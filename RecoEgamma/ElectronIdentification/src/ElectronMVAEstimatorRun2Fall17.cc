#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

ElectronMVAEstimatorRun2Fall17::ElectronMVAEstimatorRun2Fall17(const edm::ParameterSet& conf, bool withIso):
  AnyMVAEstimatorRun2Base(conf),
  tag_(conf.getParameter<std::string>("mvaTag")),
  name_(conf.getParameter<std::string>("mvaName")),
  methodName_("BDTG method"),
  beamSpotLabel_          (conf.getParameter<edm::InputTag>           ("beamSpot")),
  conversionsLabelAOD_    (conf.getParameter<edm::InputTag>           ("conversionsAOD")),
  conversionsLabelMiniAOD_(conf.getParameter<edm::InputTag>           ("conversionsMiniAOD")),
  rhoLabel_               (edm::InputTag                              ("fixedGridRhoFastjetAll")),
  ptSplit_                (conf.getParameter<double>                  ("ptSplit")),
  ebSplit_                (conf.getParameter<double>                  ("ebSplit")),
  ebeeSplit_              (conf.getParameter<double>                  ("ebeeSplit")),
  varNames_               (conf.getParameter<std::vector<std::string>>("varNames")),
  absTrackClusterMatching_(tag_ == "V2")
{

  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  std::vector<double> clipsLowerValues = conf.getParameter<std::vector<double>>("clipLower");
  std::vector<double> clipsUpperValues = conf.getParameter<std::vector<double>>("clipUpper");

  // Initialize GBRForests and and clipping instructions
  init(weightFileNames);
  setClips(clipsLowerValues, clipsUpperValues);

  withIso_ = withIso;
  debug_ = conf.getUntrackedParameter<bool>("debug", false);

}

ElectronMVAEstimatorRun2Fall17::ElectronMVAEstimatorRun2Fall17(
        const std::string &mvaTag, const std::string &mvaName, bool withIso, const std::string &conversionsTag, const std::string &beamspotTag,
        const double ptSplit, const double ebSplit, const double ebeeSplit, const bool debug):
  AnyMVAEstimatorRun2Base( edm::ParameterSet() ),
  tag_                    (mvaTag),
  name_                   (mvaName),
  methodName_             ("BDTG method"),
  beamSpotLabel_          (edm::InputTag(beamspotTag)),
  conversionsLabelAOD_    (edm::InputTag(conversionsTag)),
  conversionsLabelMiniAOD_(conversionsLabelAOD_),
  rhoLabel_               (edm::InputTag("fixedGridRhoFastjetAll")),
  ptSplit_                (ptSplit),
  ebSplit_                (ebSplit),
  ebeeSplit_              (ebeeSplit),
  absTrackClusterMatching_(tag_ == "V2")
{

  // Set if this is the ID with or without PF isolations
  withIso_ = withIso;

  // Set debug flag
  debug_ = debug;

}

void ElectronMVAEstimatorRun2Fall17::init(const std::vector<std::string> &weightFileNames) {

  // Initialize GBRForests
  if( (int)(weightFileNames.size()) != nCategories_ )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  gbrForests_.clear();
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories_; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    gbrForests_.push_back( GBRForestTools::createGBRForest( weightFile ) );

  }

}

void ElectronMVAEstimatorRun2Fall17::setClips(const std::vector<double> &clipsLowerValues, const std::vector<double> &clipsUpperValues) {

  // Set up the variable clipping intructions
  unsigned int i = 0;
  for(auto const& value: clipsLowerValues) {
      if (edm::isFinite(value)) {
          Clip clip = {i, false, (float)value};
          clipsLower_.push_back(clip);
      }
      ++i;
  }

  i = 0;
  for(auto const& value: clipsUpperValues) {
      if (edm::isFinite(value)) {
          Clip clip = {i, true, (float)value};
          clipsUpper_.push_back(clip);
      }
      ++i;
  }

}

ElectronMVAEstimatorRun2Fall17::
~ElectronMVAEstimatorRun2Fall17(){
}

void ElectronMVAEstimatorRun2Fall17::setConsumes(edm::ConsumesCollector&& cc) const {

  // All tokens for event content needed by this MVA

  // Beam spot (same for AOD and miniAOD)
  cc.consumes<reco::BeamSpot>(beamSpotLabel_);
  // Conversions collection (different names in AOD and miniAOD)
  cc.mayConsume<reco::ConversionCollection>(conversionsLabelAOD_);
  cc.mayConsume<reco::ConversionCollection>(conversionsLabelMiniAOD_);
  // Event-by-event pileup estimate rho
  cc.consumes<double>(rhoLabel_);

  //// Beam spot (same for AOD and miniAOD)
  //beamSpotToken_           = cc.consumes<reco::BeamSpot>(beamSpotLabel_);
  //// Conversions collection (different names in AOD and miniAOD)
  //conversionsTokenAOD_     = cc.mayConsume<reco::ConversionCollection>(conversionsLabelAOD_);
  //conversionsTokenMiniAOD_ = cc.mayConsume<reco::ConversionCollection>(conversionsLabelMiniAOD_);
  //// Event-by-event pileup estimate rho
  //rhoToken_                = cc.consumes<double>(rhoLabel_);

}

float ElectronMVAEstimatorRun2Fall17::
mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == nullptr ) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  }

  const int iCategory = findCategory( eleRecoPtr.get() );
  const std::vector<float> vars = fillMVAVariables( particle, iEvent );
  return mvaValue(iCategory, vars);
}

float ElectronMVAEstimatorRun2Fall17::
mvaValue( const reco::GsfElectron * particle, const edm::EventBase & iEvent) const {
  edm::Handle<reco::ConversionCollection> conversions;
  edm::Handle<reco::BeamSpot> beamSpot;
  edm::Handle<double> rho;
  iEvent.getByLabel(conversionsLabelAOD_, conversions);
  iEvent.getByLabel(beamSpotLabel_, beamSpot);
  iEvent.getByLabel(rhoLabel_, rho);
  const int iCategory = findCategory( particle );
  const std::vector<float> vars = fillMVAVariables( particle, conversions, beamSpot.product(), rho );
  return mvaValue(iCategory, vars);
}

float ElectronMVAEstimatorRun2Fall17::
mvaValue( const int iCategory, const std::vector<float> & vars) const  {
  const float result = gbrForests_.at(iCategory)->GetResponse(vars.data()); // The BDT score

  if(debug_) {
    std::cout << " *** Inside the class methodName_ " << methodName_ << std::endl;
    std::cout << " bin "                      << iCategory << std::endl
              << " see "                      << vars[0] << std::endl
              << " spp "                      << vars[1] << std::endl
              << " circularity "              << vars[2] << std::endl
              << " r9 "                       << vars[3] << std::endl
              << " etawidth "                 << vars[4] << std::endl
              << " phiwidth "                 << vars[5] << std::endl
              << " hoe "                      << vars[6] << std::endl
              << " kfhits "                   << vars[7] << std::endl
              << " kfchi2 "                   << vars[8] << std::endl
              << " gsfchi2 "                  << vars[9] << std::endl
              << " fbrem "                    << vars[10] << std::endl
              << " gsfhits "                  << vars[11] << std::endl
              << " expectedMissingInnerHits " << vars[12] << std::endl
              << " convVtxFitProbability "    << vars[13] << std::endl
              << " eop "                      << vars[14] << std::endl
              << " eleeopout "                << vars[15] << std::endl
              << " oneOverEminusOneOverP "    << vars[16] << std::endl
              << " deta "                     << vars[17] << std::endl
              << " dphi "                     << vars[18] << std::endl
              << " detacalo "                 << vars[19] << std::endl;
    if (withIso_) {
      std::cout << " ele_pfPhotonIso "          << vars[20] << std::endl
                << " ele_pfChargedHadIso "      << vars[21] << std::endl
                << " ele_pfNeutralHadIso "      << vars[22] << std::endl
                << " rho "                      << vars[23] << std::endl
                << " preShowerOverRaw "         << vars[24] << std::endl;
    }
    else {
      std::cout << " rho "                      << vars[20] << std::endl
                << " preShowerOverRaw "         << vars[21] << std::endl;
    }
    std::cout << " ### MVA " << result << std::endl << std::endl;
  }

  return result;
}

int ElectronMVAEstimatorRun2Fall17::findCategory( const edm::Ptr<reco::Candidate>& particle) const {

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == nullptr ) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  }
  return findCategory(eleRecoPtr.get());
}

int ElectronMVAEstimatorRun2Fall17::findCategory( const reco::GsfElectron * eleRecoPtr ) const {
  float pt = eleRecoPtr->pt();
  float eta = eleRecoPtr->superCluster()->eta();
  float absEta = std::abs(eta);

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;

  if (pt < ptSplit_ && absEta < ebSplit_) {
    iCategory = CAT_EB1_PTLow;
  }

  else if (pt < ptSplit_ && absEta >= ebSplit_ && absEta < ebeeSplit_) {
    iCategory = CAT_EB2_PTLow;
  }

  else if (pt < ptSplit_ && absEta >= ebeeSplit_) {
    iCategory = CAT_EE_PTLow;
  }

  else if (pt >= ptSplit_ && absEta < ebSplit_) {
    iCategory = CAT_EB1_PTHig;
  }

  else if (pt >= ptSplit_ && absEta >= ebSplit_ && absEta < ebeeSplit_) {
    iCategory = CAT_EB2_PTHig;
  }

  else if (pt >= ptSplit_ && absEta >= ebeeSplit_) {
    iCategory = CAT_EE_PTHig;
  }

  return iCategory;
}

// A function that should work on both pat and reco objects
std::vector<float> ElectronMVAEstimatorRun2Fall17::
fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {

  //
  // Declare all value maps corresponding to the products we defined earlier
  //
  edm::Handle<double> theRho;
  edm::Handle<reco::BeamSpot> theBeamSpot;
  edm::Handle<reco::ConversionCollection> conversions;

  iEvent.getByLabel(rhoLabel_, theRho);

  // Get data needed for conversion rejection
  iEvent.getByLabel(beamSpotLabel_, theBeamSpot);

  // Conversions in miniAOD and AOD have different names,
  // but the same type, so we use the same handle with different tokens.
  iEvent.getByLabel(conversionsLabelAOD_, conversions);
  if( !conversions.isValid() ) {
    iEvent.getByLabel(conversionsLabelMiniAOD_, conversions);
  }

  // Make sure everything is retrieved successfully
  if(! (theBeamSpot.isValid()
    && conversions.isValid() )
     ) {
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA"
      << std::endl
      << "Check python MVA configuration file."
      << std::endl;
  }

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == nullptr ) {
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  }

  return fillMVAVariables(eleRecoPtr.get(), conversions, theBeamSpot.product(), theRho);
}

// A function that should work on both pat and reco objects
std::vector<float> ElectronMVAEstimatorRun2Fall17::
fillMVAVariables(const reco::GsfElectron* eleRecoPtr, const edm::Handle<reco::ConversionCollection> conversions, const reco::BeamSpot *theBeamSpot, const edm::Handle<double> rho) const {

  const int iCategory = findCategory( eleRecoPtr );

  // Both pat and reco particles have exactly the same accessors, so we use a reco ptr
  // throughout the code, with a single exception as of this writing, handled separately below.
  auto superCluster = eleRecoPtr->superCluster();

  // Pure ECAL -> shower shapes
  float see            = eleRecoPtr->full5x5_sigmaIetaIeta();
  float spp            = eleRecoPtr->full5x5_sigmaIphiIphi();
  float circularity = 1. - eleRecoPtr->full5x5_e1x5() / eleRecoPtr->full5x5_e5x5();
  float r9             = eleRecoPtr->full5x5_r9();
  float etawidth       = superCluster->etaWidth();
  float phiwidth       = superCluster->phiWidth();
  float hoe            = eleRecoPtr->full5x5_hcalOverEcal(); //hadronicOverEm();
  // Endcap only variables
  float preShowerOverRaw  = superCluster->preshowerEnergy() / superCluster->rawEnergy();

  // To get to CTF track information in pat::Electron, we have to have the pointer
  // to pat::Electron, it is not accessible from the pointer to reco::GsfElectron.
  // This behavior is reported and is expected to change in the future (post-7.4.5 some time).
  bool validKF= false;
  reco::TrackRef myTrackRef = eleRecoPtr->closestCtfTrackRef();
  const pat::Electron * elePatPtr = dynamic_cast<const pat::Electron *>(eleRecoPtr);
  // Check if this is really a pat::Electron, and if yes, get the track ref from this new
  // pointer instead
  if( elePatPtr != nullptr ) {
    myTrackRef = elePatPtr->closestCtfTrackRef();
  }
  validKF = (myTrackRef.isAvailable() && (myTrackRef.isNonnull()) );

  //Pure tracking variables
  float kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  float kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  float gsfchi2         = eleRecoPtr->gsfTrack()->normalizedChi2();

  // Energy matching
  float fbrem           = eleRecoPtr->fbrem();

  float gsfhits         = eleRecoPtr->gsfTrack()->hitPattern().trackerLayersWithMeasurement();
  float expectedMissingInnerHits = eleRecoPtr->gsfTrack()
    ->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

  reco::ConversionRef convRef = ConversionTools::matchedConversion(*eleRecoPtr,
                                    conversions,
                                    theBeamSpot->position());
  float convVtxFitProbability = -1.;
  if(!convRef.isNull()) {
    const reco::Vertex &vtx = convRef.get()->conversionVertex();
    if (vtx.isValid()) {
      convVtxFitProbability = (float)TMath::Prob( vtx.chi2(), vtx.ndof());
    }
  }

  float eop             = eleRecoPtr->eSuperClusterOverP();
  float eleeopout       = eleRecoPtr->eEleClusterOverPout();
  float pAtVertex            = eleRecoPtr->trackMomentumAtVtx().R();
  float oneOverEminusOneOverP         = (1.0/eleRecoPtr->ecalEnergy()) - (1.0 / pAtVertex );

  // Geometrical matchings
  float deta            = eleRecoPtr->deltaEtaSuperClusterTrackAtVtx();
  float dphi            = eleRecoPtr->deltaPhiSuperClusterTrackAtVtx();
  float detacalo        = eleRecoPtr->deltaEtaSeedClusterTrackAtCalo();

  // Required by V2 ID
  if (absTrackClusterMatching_) {
      deta     = std::abs(deta);
      dphi     = std::abs(dphi);
      detacalo = std::abs(detacalo);
  }

  if(tag_ == "V2" && withIso_)
  {

    // Isolation variables
    float pfChargedHadIso   = (eleRecoPtr->pfIsolationVariables()).sumChargedHadronPt ; //chargedHadronIso();
    float pfNeutralHadIso   = (eleRecoPtr->pfIsolationVariables()).sumNeutralHadronEt ; //neutralHadronIso();
    float pfPhotonIso       = (eleRecoPtr->pfIsolationVariables()).sumPhotonEt; //photonIso();

    if ((iCategory + 1) % 3 == 0) {
        std::vector<float> vars = packMVAVariables(
                                      see,                      // 0
                                      spp,                      // 1
                                      circularity,
                                      r9,
                                      etawidth,
                                      phiwidth,                 // 5
                                      hoe,
                                      //Pure tracking variables
                                      kfhits,
                                      kfchi2,
                                      gsfchi2,                  // 9
                                      // Energy matching
                                      fbrem,
                                      gsfhits,
                                      expectedMissingInnerHits,
                                      convVtxFitProbability,     // 13
                                      eop,
                                      eleeopout,                // 15
                                      oneOverEminusOneOverP,
                                      // Geometrical matchings
                                      deta,                     // 17
                                      dphi,
                                      detacalo,
                                      // Pileup
                                      (float)*rho,
                                      // Endcap only
                                      preShowerOverRaw,         // 24
                                      // Isolation variables
                                      pfPhotonIso,
                                      pfChargedHadIso,
                                      pfNeutralHadIso
                                  );
        constrainMVAVariables(vars);
        return vars;
    } else {
        std::vector<float> vars = packMVAVariables(
                                      see,                      // 0
                                      spp,                      // 1
                                      circularity,
                                      r9,
                                      etawidth,
                                      phiwidth,                 // 5
                                      hoe,
                                      //Pure tracking variables
                                      kfhits,
                                      kfchi2,
                                      gsfchi2,                  // 9
                                      // Energy matching
                                      fbrem,
                                      gsfhits,
                                      expectedMissingInnerHits,
                                      convVtxFitProbability,     // 13
                                      eop,
                                      eleeopout,                // 15
                                      oneOverEminusOneOverP,
                                      // Geometrical matchings
                                      deta,                     // 17
                                      dphi,
                                      detacalo,
                                      // Pileup
                                      (float)*rho,
                                      // Isolation variables
                                      pfPhotonIso,
                                      pfChargedHadIso,
                                      pfNeutralHadIso
                                  );
        constrainMVAVariables(vars);
        return vars;
    }
  }
  else if(withIso_)
  {

    // Isolation variables
    float pfChargedHadIso   = (eleRecoPtr->pfIsolationVariables()).sumChargedHadronPt ; //chargedHadronIso();
    float pfNeutralHadIso   = (eleRecoPtr->pfIsolationVariables()).sumNeutralHadronEt ; //neutralHadronIso();
    float pfPhotonIso       = (eleRecoPtr->pfIsolationVariables()).sumPhotonEt; //photonIso();

    /*
     * Packing variables for the MVA evaluation.
     * CAREFUL: It is critical that all the variables that are packed into “vars” are
     * exactly in the order they are found in the weight files
     */
    std::vector<float> vars = packMVAVariables(
                                  see,                      // 0
                                  spp,                      // 1
                                  circularity,
                                  r9,
                                  etawidth,
                                  phiwidth,                 // 5
                                  hoe,
                                  //Pure tracking variables
                                  kfhits,
                                  kfchi2,
                                  gsfchi2,                  // 9
                                  // Energy matching
                                  fbrem,
                                  gsfhits,
                                  expectedMissingInnerHits,
                                  convVtxFitProbability,     // 13
                                  eop,
                                  eleeopout,                // 15
                                  oneOverEminusOneOverP,
                                  // Geometrical matchings
                                  deta,                     // 17
                                  dphi,
                                  detacalo,
                                  // Isolation variables
                                  pfPhotonIso,
                                  pfChargedHadIso,
                                  pfNeutralHadIso,
                                  // Pileup
                                  (float)*rho,

                                  // Endcap only variables NOTE: we don't need
                                  // to check if we are actually in the endcap
                                  // or not, as it is the last variable in the
                                  // vector and it will be ignored by the
                                  // GBRForest for the barrel.
                                  //
                                  // The GBRForest classification just needs an
                                  // array with the input variables in the
                                  // right order, what comes after doesn't
                                  // matter.
                                  preShowerOverRaw          // 24
                              );

    constrainMVAVariables(vars);

    return vars;
  }
  else
  {
    std::vector<float> vars = packMVAVariables(
                                  see,                      // 0
                                  spp,                      // 1
                                  circularity,
                                  r9,
                                  etawidth,
                                  phiwidth,                 // 5
                                  hoe,
                                  kfhits,
                                  kfchi2,
                                  gsfchi2,                  // 9
                                  fbrem,
                                  gsfhits,
                                  expectedMissingInnerHits,
                                  convVtxFitProbability,     // 13
                                  eop,
                                  eleeopout,                // 15
                                  oneOverEminusOneOverP,
                                  deta,                     // 17
                                  dphi,
                                  detacalo,
                                  (float)*rho,
                                  preShowerOverRaw          // 21
                              );

    constrainMVAVariables(vars);

    return vars;
  }
}

void ElectronMVAEstimatorRun2Fall17::constrainMVAVariables(std::vector<float>& vars) const {

  // Check that variables do not have crazy values

  for(auto const& clip: clipsLower_) {
      if ( vars[clip.varIdx] < clip.value  ) {
          vars[clip.varIdx] =   clip.value;
      }
  }

  for(auto const& clip: clipsUpper_) {
      if ( vars[clip.varIdx] > clip.value  ) {
          vars[clip.varIdx] =   clip.value;
      }
  }

}
