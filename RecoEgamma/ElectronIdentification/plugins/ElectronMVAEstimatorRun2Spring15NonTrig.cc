#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Spring15NonTrig.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMath.h"
#include "TMVA/MethodBDT.h"

ElectronMVAEstimatorRun2Spring15NonTrig::ElectronMVAEstimatorRun2Spring15NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  _tag(conf.getParameter<std::string>("mvaTag")),
  _MethodName("BDTG method"),
  _beamSpotLabel(conf.getParameter<edm::InputTag>("beamSpot")),
  _conversionsLabelAOD(conf.getParameter<edm::InputTag>("conversionsAOD")),
  _conversionsLabelMiniAOD(conf.getParameter<edm::InputTag>("conversionsMiniAOD")) {
  
  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  if( (int)(weightFileNames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _gbrForests.clear();
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    _gbrForests.push_back( GBRForestTools::createGBRForest( weightFile ) );

  }

}

ElectronMVAEstimatorRun2Spring15NonTrig::
~ElectronMVAEstimatorRun2Spring15NonTrig(){
}


void ElectronMVAEstimatorRun2Spring15NonTrig::setConsumes(edm::ConsumesCollector&& cc) const {

  // All tokens for event content needed by this MVA

  // Beam spot (same for AOD and miniAOD)
   cc.consumes<reco::BeamSpot>(_beamSpotLabel);

  // Conversions collection (different names in AOD and miniAOD)
  cc.mayConsume<reco::ConversionCollection>(_conversionsLabelAOD);
  cc.mayConsume<reco::ConversionCollection>(_conversionsLabelMiniAOD);
  

}

float ElectronMVAEstimatorRun2Spring15NonTrig::
mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  
  const int iCategory = findCategory( particle );
  const std::vector<float> vars = fillMVAVariables( particle, iEvent );
  const float result = _gbrForests.at(iCategory)->GetResponse(vars.data()); // The BDT score

  const bool debug = false;
  if(debug) {
    std::cout << " *** Inside the class _MethodName " << _MethodName << std::endl;
    std::cout << " bin " << iCategory
	      << " fbrem " <<  vars[11]
	      << " kfchi2 " << vars[9]
	      << " mykfhits " << vars[8]
	      << " gsfchi2 " << vars[10]
	      << " deta " <<  vars[18]
	      << " dphi " << vars[19]
	      << " detacalo " << vars[20]
	      << " see " << vars[0]
	      << " spp " << vars[1]
	      << " etawidth " << vars[4]
	      << " phiwidth " << vars[5]
	      << " OneMinusE1x5E5x5 " << vars[2]
	      << " R9 " << vars[3]
	      << " HoE " << vars[6]
	      << " EoP " << vars[15]
	      << " IoEmIoP " << vars[17]
	      << " eleEoPout " << vars[16]
	      << " eta " << vars[24]
	      << " pt " << vars[21] << std::endl;
    std::cout << " ### MVA " << result << std::endl;
  }

  return result;
}

int ElectronMVAEstimatorRun2Spring15NonTrig::findCategory( const edm::Ptr<reco::Candidate>& particle) const {
  
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == nullptr )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;

  float pt = eleRecoPtr->pt();
  float eta = eleRecoPtr->superCluster()->eta();

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;
  const float ptSplit = 10;   // we have above and below 10 GeV categories
  const float ebSplit = 0.800;// barrel is split into two regions
  const float ebeeSplit = 1.479; // division between barrel and endcap

  if (pt < ptSplit && std::abs(eta) < ebSplit)  
    iCategory = CAT_EB1_PT5to10;

  if (pt < ptSplit && std::abs(eta) >= ebSplit && std::abs(eta) < ebeeSplit)
    iCategory = CAT_EB2_PT5to10;

  if (pt < ptSplit && std::abs(eta) >= ebeeSplit) 
    iCategory = CAT_EE_PT5to10;

  if (pt >= ptSplit && std::abs(eta) < ebSplit) 
    iCategory = CAT_EB1_PT10plus;

  if (pt >= ptSplit && std::abs(eta) >= ebSplit && std::abs(eta) < ebeeSplit)
    iCategory = CAT_EB2_PT10plus;

  if (pt >= ptSplit && std::abs(eta) >= ebeeSplit) 
    iCategory = CAT_EE_PT10plus;
  
  return iCategory;
}

bool ElectronMVAEstimatorRun2Spring15NonTrig::
isEndcapCategory(int category ) const {

  bool isEndcap = false;
  if( category == CAT_EE_PT5to10 || category == CAT_EE_PT10plus )
    isEndcap = true;

  return isEndcap;
}

// A function that should work on both pat and reco objects
std::vector<float> ElectronMVAEstimatorRun2Spring15NonTrig::
fillMVAVariables(const edm::Ptr<reco::Candidate>& particle,
                 const edm::Event& iEvent ) const {

  // 
  // Declare all value maps corresponding to the products we defined earlier
  //
  edm::Handle<reco::BeamSpot> theBeamSpot;
  edm::Handle<reco::ConversionCollection> conversions;

  // Get data needed for conversion rejection
  iEvent.getByLabel(_beamSpotLabel, theBeamSpot);

  // Conversions in miniAOD and AOD have different names,
  // but the same type, so we use the same handle with different tokens.
  iEvent.getByLabel(_conversionsLabelAOD, conversions);
  if( !conversions.isValid() )
    iEvent.getByLabel(_conversionsLabelMiniAOD, conversions);

  // Make sure everything is retrieved successfully
  if(! (theBeamSpot.isValid() 
	&& conversions.isValid() ) 
     )
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA" 
      << std::endl
      << "Check python MVA configuration file."
      << std::endl;

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == nullptr )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;

  // Both pat and reco particles have exactly the same accessors, so we use a reco ptr 
  // throughout the code, with a single exception as of this writing, handled separately below.
  auto superCluster = eleRecoPtr->superCluster();
  
  AllVariables allMVAVars;

  // Pure ECAL -> shower shapes
  allMVAVars.see            = eleRecoPtr->full5x5_sigmaIetaIeta();
  allMVAVars.spp            = eleRecoPtr->full5x5_sigmaIphiIphi();
  allMVAVars.OneMinusE1x5E5x5 = 1. - eleRecoPtr->full5x5_e1x5() / eleRecoPtr->full5x5_e5x5();
  allMVAVars.R9             = eleRecoPtr->full5x5_r9();
  allMVAVars.etawidth       = superCluster->etaWidth();
  allMVAVars.phiwidth       = superCluster->phiWidth();
  allMVAVars.HoE            = eleRecoPtr->hadronicOverEm();
  // Endcap only variables
  allMVAVars.PreShowerOverRaw  = superCluster->preshowerEnergy() / superCluster->rawEnergy();

  // To get to CTF track information in pat::Electron, we have to have the pointer
  // to pat::Electron, it is not accessible from the pointer to reco::GsfElectron.
  // This behavior is reported and is expected to change in the future (post-7.4.5 some time).
  bool validKF= false; 
  reco::TrackRef myTrackRef = eleRecoPtr->closestCtfTrackRef();
  const edm::Ptr<pat::Electron> elePatPtr(eleRecoPtr);
  // Check if this is really a pat::Electron, and if yes, get the track ref from this new
  // pointer instead
  if( elePatPtr.get() != nullptr )
    myTrackRef = elePatPtr->closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable() && (myTrackRef.isNonnull()) );  

  //Pure tracking variables
  allMVAVars.kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  allMVAVars.kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  allMVAVars.gsfchi2         = eleRecoPtr->gsfTrack()->normalizedChi2();

  // Energy matching
  allMVAVars.fbrem           = eleRecoPtr->fbrem();

  allMVAVars.gsfhits         = eleRecoPtr->gsfTrack()->hitPattern().trackerLayersWithMeasurement();
  allMVAVars.expectedMissingInnerHits = eleRecoPtr->gsfTrack()
    ->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

  reco::ConversionRef conv_ref = ConversionTools::matchedConversion(*eleRecoPtr,
								    conversions, 
								    theBeamSpot->position());
  double vertexFitProbability = -1.; 
  if(!conv_ref.isNull()) {
    const reco::Vertex &vtx = conv_ref.get()->conversionVertex(); if (vtx.isValid()) {
      vertexFitProbability = TMath::Prob( vtx.chi2(), vtx.ndof());
    } 
  }
  allMVAVars.convVtxFitProbability    = vertexFitProbability;

  allMVAVars.EoP             = eleRecoPtr->eSuperClusterOverP();
  allMVAVars.eleEoPout       = eleRecoPtr->eEleClusterOverPout();
  float pAtVertex            = eleRecoPtr->trackMomentumAtVtx().R();
  allMVAVars.IoEmIoP         = (1.0/eleRecoPtr->ecalEnergy()) - (1.0 / pAtVertex );

  // Geometrical matchings
  allMVAVars.deta            = eleRecoPtr->deltaEtaSuperClusterTrackAtVtx();
  allMVAVars.dphi            = eleRecoPtr->deltaPhiSuperClusterTrackAtVtx();
  allMVAVars.detacalo        = eleRecoPtr->deltaEtaSeedClusterTrackAtCalo();

  // Spectator variables  
  allMVAVars.pt              = eleRecoPtr->pt();
  float scEta = superCluster->eta();
  constexpr float ebeeSplit = 1.479;
  allMVAVars.isBarrel        = ( std::abs(scEta) < ebeeSplit );
  allMVAVars.isEndcap        = ( std::abs(scEta) >= ebeeSplit );
  allMVAVars.SCeta           = scEta;
  // The spectator variables below were examined for training, but
  // are not necessary for evaluating the discriminator, so they are
  // given dummy values (the specator variables above are also unimportant).
  // They are introduced only to match the definition of the discriminator 
  // in the weights file.
  constexpr unsigned nines = 999;
  allMVAVars.eClass               = nines;
  allMVAVars.pfRelIso             = nines;
  allMVAVars.expectedInnerHits    = nines;
  allMVAVars.vtxconv              = nines;
  allMVAVars.mcEventWeight        = nines;
  allMVAVars.mcCBmatchingCategory = nines;

  constrainMVAVariables(allMVAVars);

  std::vector<float> vars;

  if( isEndcapCategory( findCategory( particle ) ) ) {
    vars = packMVAVariables(allMVAVars.see,
                                       allMVAVars.spp,
                                       allMVAVars.OneMinusE1x5E5x5,
                                       allMVAVars.R9,
                                       allMVAVars.etawidth,
                                       allMVAVars.phiwidth,
                                       allMVAVars.HoE,
                                       // Endcap only variables
                                       allMVAVars.PreShowerOverRaw,                                       
                                       //Pure tracking variables
                                       allMVAVars.kfhits,
                                       allMVAVars.kfchi2,
                                       allMVAVars.gsfchi2,                                       
                                       // Energy matching
                                       allMVAVars.fbrem,                                       
                                       allMVAVars.gsfhits,
                                       allMVAVars.expectedMissingInnerHits,
                                       allMVAVars.convVtxFitProbability,
                                       allMVAVars.EoP,
                                       allMVAVars.eleEoPout,
                                       allMVAVars.IoEmIoP,                                       
                                       // Geometrical matchings
                                       allMVAVars.deta,
                                       allMVAVars.dphi,
                                       allMVAVars.detacalo,                                       
                                       // Spectator variables  
                                       allMVAVars.pt,
                                       allMVAVars.isBarrel,
                                       allMVAVars.isEndcap,
                                       allMVAVars.SCeta,                                       
                                       allMVAVars.eClass,
                                       allMVAVars.pfRelIso,
                                       allMVAVars.expectedInnerHits,
                                       allMVAVars.vtxconv,
                                       allMVAVars.mcEventWeight,
                                       allMVAVars.mcCBmatchingCategory);
  } else {
    vars = packMVAVariables(allMVAVars.see,
                                       allMVAVars.spp,
                                       allMVAVars.OneMinusE1x5E5x5,
                                       allMVAVars.R9,
                                       allMVAVars.etawidth,
                                       allMVAVars.phiwidth,
                                       allMVAVars.HoE,
                                       //Pure tracking variables
                                       allMVAVars.kfhits,
                                       allMVAVars.kfchi2,
                                       allMVAVars.gsfchi2,                                       
                                       // Energy matching
                                       allMVAVars.fbrem,                                       
                                       allMVAVars.gsfhits,
                                       allMVAVars.expectedMissingInnerHits,
                                       allMVAVars.convVtxFitProbability,
                                       allMVAVars.EoP,
                                       allMVAVars.eleEoPout,
                                       allMVAVars.IoEmIoP,                                       
                                       // Geometrical matchings
                                       allMVAVars.deta,
                                       allMVAVars.dphi,
                                       allMVAVars.detacalo,                                       
                                       // Spectator variables  
                                       allMVAVars.pt,
                                       allMVAVars.isBarrel,
                                       allMVAVars.isEndcap,
                                       allMVAVars.SCeta,                                       
                                       allMVAVars.eClass,
                                       allMVAVars.pfRelIso,
                                       allMVAVars.expectedInnerHits,
                                       allMVAVars.vtxconv,
                                       allMVAVars.mcEventWeight,
                                       allMVAVars.mcCBmatchingCategory);
  }
  return vars;
}

void ElectronMVAEstimatorRun2Spring15NonTrig::constrainMVAVariables(AllVariables& allMVAVars) const {

  // Check that variables do not have crazy values

  if(allMVAVars.fbrem < -1.)
    allMVAVars.fbrem = -1.;
  
  allMVAVars.deta = fabs(allMVAVars.deta);
  if(allMVAVars.deta > 0.06)
    allMVAVars.deta = 0.06;
  
  
  allMVAVars.dphi = fabs(allMVAVars.dphi);
  if(allMVAVars.dphi > 0.6)
    allMVAVars.dphi = 0.6;
  

  if(allMVAVars.EoP > 20.)
    allMVAVars.EoP = 20.;
  
  if(allMVAVars.eleEoPout > 20.)
    allMVAVars.eleEoPout = 20.;
  
  
  allMVAVars.detacalo = fabs(allMVAVars.detacalo);
  if(allMVAVars.detacalo > 0.2)
    allMVAVars.detacalo = 0.2;
  
  if(allMVAVars.OneMinusE1x5E5x5 < -1.)
    allMVAVars.OneMinusE1x5E5x5 = -1;
  
  if(allMVAVars.OneMinusE1x5E5x5 > 2.)
    allMVAVars.OneMinusE1x5E5x5 = 2.; 
  
  
  
  if(allMVAVars.R9 > 5)
    allMVAVars.R9 = 5;
  
  if(allMVAVars.gsfchi2 > 200.)
    allMVAVars.gsfchi2 = 200;
  
  
  if(allMVAVars.kfchi2 > 10.)
    allMVAVars.kfchi2 = 10.;
  

}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  ElectronMVAEstimatorRun2Spring15NonTrig,
		  "ElectronMVAEstimatorRun2Spring15NonTrig");
