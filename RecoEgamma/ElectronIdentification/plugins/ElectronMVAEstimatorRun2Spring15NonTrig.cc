#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Spring15NonTrig.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMath.h"

ElectronMVAEstimatorRun2Spring15NonTrig::ElectronMVAEstimatorRun2Spring15NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf){

  _tag = conf.getParameter<std::string>("mvaTag");
  
  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  if( (int)(weightFileNames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _tmvaReaders.clear();
  _MethodName = "BDTG method";
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    _tmvaReaders.push_back( std::unique_ptr<TMVA::Reader> ( createSingleReader(i, weightFile ) ) );

  }

}

ElectronMVAEstimatorRun2Spring15NonTrig::
~ElectronMVAEstimatorRun2Spring15NonTrig(){
  
  _tmvaReaders.clear();
}


void ElectronMVAEstimatorRun2Spring15NonTrig::setConsumes(edm::ConsumesCollector&& cc){

  // All tokens for event content needed by this MVA

  // Beam spot (same for AOD and miniAOD)
  _beamSpotToken = cc.consumes < reco::BeamSpot >
    (_conf.getParameter<edm::InputTag>("beamSpot"));

  // Conversions collection (different names in AOD and miniAOD)
  _conversionsTokenAOD = cc.mayConsume < reco::ConversionCollection >
    (_conf.getParameter<edm::InputTag>("conversionsAOD"));
  _conversionsTokenMiniAOD = cc.mayConsume < reco::ConversionCollection >
    (_conf.getParameter<edm::InputTag>("conversionsMiniAOD"));
  

}

void ElectronMVAEstimatorRun2Spring15NonTrig::getEventContent(const edm::Event& iEvent){

  // Get data needed for conversion rejection
  iEvent.getByToken(_beamSpotToken, _theBeamSpot);

  // Conversions in miniAOD and AOD have different names,
  // but the same type, so we use the same handle with different tokens.
  iEvent.getByToken(_conversionsTokenAOD, _conversions);
  if( !_conversions.isValid() )
    iEvent.getByToken(_conversionsTokenMiniAOD, _conversions);

  // Make sure everything is retrieved successfully
  if(! (_theBeamSpot.isValid() 
	&& _conversions.isValid() ) 
     )
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA" 
      << std::endl
      << "Check python MVA configuration file."
      << std::endl;
  
}

float ElectronMVAEstimatorRun2Spring15NonTrig::
mvaValue( const edm::Ptr<reco::Candidate>& particle){
  
  int iCategory = findCategory( particle );
  fillMVAVariables( particle );  
  constrainMVAVariables();
  float result = _tmvaReaders.at(iCategory)->EvaluateMVA(_MethodName);

  bool debug = false;
  if(debug) {
    std::cout << " *** Inside the class _MethodName " << _MethodName << std::endl;
    std::cout << " bin " << iCategory
	      << " fbrem " <<  _allMVAVars.fbrem
	      << " kfchi2 " << _allMVAVars.kfchi2
	      << " mykfhits " << _allMVAVars.kfhits
	      << " gsfchi2 " << _allMVAVars.gsfchi2
	      << " deta " <<  _allMVAVars.deta
	      << " dphi " << _allMVAVars.dphi
	      << " detacalo " << _allMVAVars.detacalo
	      << " see " << _allMVAVars.see
	      << " spp " << _allMVAVars.spp
	      << " etawidth " << _allMVAVars.etawidth
	      << " phiwidth " << _allMVAVars.phiwidth
	      << " OneMinusE1x5E5x5 " << _allMVAVars.OneMinusE1x5E5x5
	      << " R9 " << _allMVAVars.R9
	      << " HoE " << _allMVAVars.HoE
	      << " EoP " << _allMVAVars.EoP
	      << " IoEmIoP " << _allMVAVars.IoEmIoP
	      << " eleEoPout " << _allMVAVars.eleEoPout
	      << " eta " << _allMVAVars.SCeta
	      << " pt " << _allMVAVars.pt << std::endl;
    std::cout << " ### MVA " << result << std::endl;
  }

  return result;
}

int ElectronMVAEstimatorRun2Spring15NonTrig::findCategory( const edm::Ptr<reco::Candidate>& particle){
  
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == NULL )
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
isEndcapCategory(int category ){

  bool isEndcap = false;
  if( category == CAT_EE_PT5to10 || category == CAT_EE_PT10plus )
    isEndcap = true;

  return isEndcap;
}


TMVA::Reader *ElectronMVAEstimatorRun2Spring15NonTrig::
createSingleReader(const int iCategory, const edm::FileInPath &weightFile){

  //
  // Create the reader  
  //
  TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:Silent:!Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //
  // Pure ECAL -> shower shapes
  tmpTMVAReader->AddVariable("ele_oldsigmaietaieta", &_allMVAVars.see);
  tmpTMVAReader->AddVariable("ele_oldsigmaiphiiphi", &_allMVAVars.spp);
  tmpTMVAReader->AddVariable("ele_oldcircularity",   &_allMVAVars.OneMinusE1x5E5x5);
  tmpTMVAReader->AddVariable("ele_oldr9",            &_allMVAVars.R9);
  tmpTMVAReader->AddVariable("ele_scletawidth",      &_allMVAVars.etawidth);
  tmpTMVAReader->AddVariable("ele_sclphiwidth",      &_allMVAVars.phiwidth);
  tmpTMVAReader->AddVariable("ele_he",               &_allMVAVars.HoE);
  // Endcap only variables
  if( isEndcapCategory(iCategory) )
    tmpTMVAReader->AddVariable("ele_psEoverEraw",    &_allMVAVars.PreShowerOverRaw);
  
  //Pure tracking variables
  tmpTMVAReader->AddVariable("ele_kfhits",           &_allMVAVars.kfhits);
  tmpTMVAReader->AddVariable("ele_kfchi2",           &_allMVAVars.kfchi2);
  tmpTMVAReader->AddVariable("ele_gsfchi2",        &_allMVAVars.gsfchi2);

  // Energy matching
  tmpTMVAReader->AddVariable("ele_fbrem",           &_allMVAVars.fbrem);

  tmpTMVAReader->AddVariable("ele_gsfhits",         &_allMVAVars.gsfhits);
  tmpTMVAReader->AddVariable("ele_expected_inner_hits",             &_allMVAVars.expectedMissingInnerHits);
  tmpTMVAReader->AddVariable("ele_conversionVertexFitProbability",  &_allMVAVars.convVtxFitProbability);

  tmpTMVAReader->AddVariable("ele_ep",              &_allMVAVars.EoP);
  tmpTMVAReader->AddVariable("ele_eelepout",        &_allMVAVars.eleEoPout);
  tmpTMVAReader->AddVariable("ele_IoEmIop",         &_allMVAVars.IoEmIoP);
  
  // Geometrical matchings
  tmpTMVAReader->AddVariable("ele_deltaetain",      &_allMVAVars.deta);
  tmpTMVAReader->AddVariable("ele_deltaphiin",      &_allMVAVars.dphi);
  tmpTMVAReader->AddVariable("ele_deltaetaseed",    &_allMVAVars.detacalo);
  
  // Spectator variables  
  tmpTMVAReader->AddSpectator("ele_pT",             &_allMVAVars.pt);
  tmpTMVAReader->AddSpectator("ele_isbarrel",       &_allMVAVars.isBarrel);
  tmpTMVAReader->AddSpectator("ele_isendcap",       &_allMVAVars.isEndcap);
  tmpTMVAReader->AddSpectator("scl_eta",            &_allMVAVars.SCeta);

  tmpTMVAReader->AddSpectator("ele_eClass",                 &_allMVAVars.eClass);
  tmpTMVAReader->AddSpectator("ele_pfRelIso",               &_allMVAVars.pfRelIso);
  tmpTMVAReader->AddSpectator("ele_expected_inner_hits",    &_allMVAVars.expectedInnerHits);
  tmpTMVAReader->AddSpectator("ele_vtxconv",                &_allMVAVars.vtxconv);
  tmpTMVAReader->AddSpectator("mc_event_weight",            &_allMVAVars.mcEventWeight);
  tmpTMVAReader->AddSpectator("mc_ele_CBmatching_category", &_allMVAVars.mcCBmatchingCategory);

  //
  // Book the method and set up the weights file
  //
  tmpTMVAReader->BookMVA(_MethodName , weightFile.fullPath() );

  return tmpTMVAReader;
}

// A function that should work on both pat and reco objects
void ElectronMVAEstimatorRun2Spring15NonTrig::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle){

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == NULL )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;

  // Both pat and reco particles have exactly the same accessors, so we use a reco ptr 
  // throughout the code, with a single exception as of this writing, handled separately below.
  auto superCluster = eleRecoPtr->superCluster();

  // Pure ECAL -> shower shapes
  _allMVAVars.see            = eleRecoPtr->full5x5_sigmaIetaIeta();
  _allMVAVars.spp            = eleRecoPtr->full5x5_sigmaIphiIphi();
  _allMVAVars.OneMinusE1x5E5x5 = 1. - eleRecoPtr->full5x5_e1x5() / eleRecoPtr->full5x5_e5x5();
  _allMVAVars.R9             = eleRecoPtr->full5x5_r9();
  _allMVAVars.etawidth       = superCluster->etaWidth();
  _allMVAVars.phiwidth       = superCluster->phiWidth();
  _allMVAVars.HoE            = eleRecoPtr->hadronicOverEm();
  // Endcap only variables
  _allMVAVars.PreShowerOverRaw  = superCluster->preshowerEnergy() / superCluster->rawEnergy();

  // To get to CTF track information in pat::Electron, we have to have the pointer
  // to pat::Electron, it is not accessible from the pointer to reco::GsfElectron.
  // This behavior is reported and is expected to change in the future (post-7.4.5 some time).
  bool validKF= false; 
  reco::TrackRef myTrackRef = eleRecoPtr->closestCtfTrackRef();
  const edm::Ptr<pat::Electron> elePatPtr(eleRecoPtr);
  // Check if this is really a pat::Electron, and if yes, get the track ref from this new
  // pointer instead
  if( elePatPtr.get() != NULL )
    myTrackRef = elePatPtr->closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable() && (myTrackRef.isNonnull()) );  

  //Pure tracking variables
  _allMVAVars.kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  _allMVAVars.kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  _allMVAVars.gsfchi2         = eleRecoPtr->gsfTrack()->normalizedChi2();

  // Energy matching
  _allMVAVars.fbrem           = eleRecoPtr->fbrem();

  _allMVAVars.gsfhits         = eleRecoPtr->gsfTrack()->found();
  _allMVAVars.expectedMissingInnerHits = eleRecoPtr->gsfTrack()
    ->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS);

  reco::ConversionRef conv_ref = ConversionTools::matchedConversion(*eleRecoPtr,
								    _conversions, 
								    _theBeamSpot->position());
  double vertexFitProbability = -1.; 
  if(!conv_ref.isNull()) {
    const reco::Vertex &vtx = conv_ref.get()->conversionVertex(); if (vtx.isValid()) {
      vertexFitProbability = TMath::Prob( vtx.chi2(), vtx.ndof());
    } 
  }
  _allMVAVars.convVtxFitProbability    = vertexFitProbability;

  _allMVAVars.EoP             = eleRecoPtr->eSuperClusterOverP();
  _allMVAVars.eleEoPout       = eleRecoPtr->eEleClusterOverPout();
  _allMVAVars.IoEmIoP         = (1.0/eleRecoPtr->ecalEnergy()) - (1.0 / eleRecoPtr->p());

  // Geometrical matchings
  _allMVAVars.deta            = eleRecoPtr->deltaEtaSuperClusterTrackAtVtx();
  _allMVAVars.dphi            = eleRecoPtr->deltaPhiSuperClusterTrackAtVtx();
  _allMVAVars.detacalo        = eleRecoPtr->deltaEtaSeedClusterTrackAtCalo();

  // Spectator variables  
  _allMVAVars.pt              = eleRecoPtr->pt();
  float scEta = superCluster->eta();
  _allMVAVars.isBarrel        = ( std::abs(scEta) < 1.479 );
  _allMVAVars.isEndcap        = ( std::abs(scEta) >= 1.479);
  _allMVAVars.SCeta           = scEta;
  // The spectator variables below were examined for training, but
  // are not necessary for evaluating the discriminator, so they are
  // given dummy values (the specator variables above are also unimportant).
  // They are introduced only to match the definition of the discriminator 
  // in the weights file.
  _allMVAVars.eClass               = 999;
  _allMVAVars.pfRelIso             = 999;
  _allMVAVars.expectedInnerHits    = 999;
  _allMVAVars.vtxconv              = 999;
  _allMVAVars.mcEventWeight        = 999;
  _allMVAVars.mcCBmatchingCategory = 999;


}

void ElectronMVAEstimatorRun2Spring15NonTrig::constrainMVAVariables(){

  // Check that variables do not have crazy values

  if(_allMVAVars.fbrem < -1.)
    _allMVAVars.fbrem = -1.;
  
  _allMVAVars.deta = fabs(_allMVAVars.deta);
  if(_allMVAVars.deta > 0.06)
    _allMVAVars.deta = 0.06;
  
  
  _allMVAVars.dphi = fabs(_allMVAVars.dphi);
  if(_allMVAVars.dphi > 0.6)
    _allMVAVars.dphi = 0.6;
  

  if(_allMVAVars.EoP > 20.)
    _allMVAVars.EoP = 20.;
  
  if(_allMVAVars.eleEoPout > 20.)
    _allMVAVars.eleEoPout = 20.;
  
  
  _allMVAVars.detacalo = fabs(_allMVAVars.detacalo);
  if(_allMVAVars.detacalo > 0.2)
    _allMVAVars.detacalo = 0.2;
  
  if(_allMVAVars.OneMinusE1x5E5x5 < -1.)
    _allMVAVars.OneMinusE1x5E5x5 = -1;
  
  if(_allMVAVars.OneMinusE1x5E5x5 > 2.)
    _allMVAVars.OneMinusE1x5E5x5 = 2.; 
  
  
  
  if(_allMVAVars.R9 > 5)
    _allMVAVars.R9 = 5;
  
  if(_allMVAVars.gsfchi2 > 200.)
    _allMVAVars.gsfchi2 = 200;
  
  
  if(_allMVAVars.kfchi2 > 10.)
    _allMVAVars.kfchi2 = 10.;
  

}

