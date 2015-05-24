#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Phys14NonTrig.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

ElectronMVAEstimatorRun2Phys14NonTrig::ElectronMVAEstimatorRun2Phys14NonTrig( std::vector<std::string> filenames){

  if( (int)(filenames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _tmvaReaders.clear();
  _MethodName = "BDTG method";
  for(int i=0; i<nCategories; i++){
    
    TMVA::Reader *thisReader = createSingleReader(i, filenames.at(i) ) ;
    _tmvaReaders.push_back( thisReader );

  }

}

ElectronMVAEstimatorRun2Phys14NonTrig::
~ElectronMVAEstimatorRun2Phys14NonTrig(){
  
  // It is expected that as vector clears its contents,
  // the delete is called on each pointer automatically
  _tmvaReaders.clear();
}

float ElectronMVAEstimatorRun2Phys14NonTrig::
mvaValue( edm::Ptr<reco::Candidate>& particle){

  //
  // Try to cast the particle into a pat (first) or reco particle
  edm::Ptr<pat::Electron> elePat = ( edm::Ptr<pat::Electron> )particle;
  edm::Ptr<reco::GsfElectron> eleReco = ( edm::Ptr<reco::GsfElectron> )particle;
  
  int iCategory = UNDEFINED;

  if( !elePat.isNull() ){
    // cast is successful
    iCategory = findCategory( elePat );
    fillMVAVariables( elePat );
  }else if ( ! eleReco.isNull() ){
    // Normally a pat particle can be cast into reco particle too,
    // that's why reco goes second
    iCategory = findCategory( eleReco );
    fillMVAVariables( eleReco );
  }else
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;
  
  float result = _tmvaReaders.at(iCategory)->EvaluateMVA(_MethodName);

  return result;
}

template <typename T> 
int ElectronMVAEstimatorRun2Phys14NonTrig::
findCategory( const T& particle){
  
  float pt = particle->pt();
  float eta = particle->superCluster()->eta();

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;
  const float ptSplit = 10;   // we have above and below 10 GeV categories
  const float ebSplit = 0.800;// barrel is split into two regions
  const float ebeeSplit = 1.479; // division between barrel and endcap

  if (pt < ptSplit && fabs(eta) < ebSplit)  
    iCategory = CAT_EB1_PT5to10;

  if (pt < ptSplit && fabs(eta) >= ebSplit && fabs(eta) < ebeeSplit)
    iCategory = CAT_EB2_PT5to10;

  if (pt < ptSplit && fabs(eta) >= ebeeSplit) 
    iCategory = CAT_EE_PT5to10;

  if (pt >= ptSplit && fabs(eta) < ebSplit) 
    iCategory = CAT_EB1_PT10plus;

  if (pt >= ptSplit && fabs(eta) >= ebSplit && fabs(eta) < ebeeSplit)
    iCategory = CAT_EB2_PT10plus;

  if (pt >= ptSplit && fabs(eta) >= ebeeSplit) 
    iCategory = CAT_EE_PT10plus;
  
  return iCategory;
}

bool ElectronMVAEstimatorRun2Phys14NonTrig::
isEndcapCategory(int category ){

  bool isEndcap = false;
  if( category == CAT_EE_PT5to10 || category == CAT_EE_PT10plus )
    isEndcap = true;

  return isEndcap;
}


TMVA::Reader *ElectronMVAEstimatorRun2Phys14NonTrig::
createSingleReader(int iCategory, std::string filename){

  //
  // Create the reader  
  //
  TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:!Silent:Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //

  tmpTMVAReader->AddVariable("ele_kfhits",           &_allMVAVars.kfhits);
  
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
  tmpTMVAReader->AddVariable("ele_kfchi2",           &_allMVAVars.kfchi2);
  tmpTMVAReader->AddVariable("ele_chi2_hits",        &_allMVAVars.gsfchi2);

  // Energy matching
  tmpTMVAReader->AddVariable("ele_fbrem",           &_allMVAVars.fbrem);
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

  //
  // Book the method and set up the weights file
  //
  edm::FileInPath weightsFile( filename );
  tmpTMVAReader->BookMVA(_MethodName , filename);

  return tmpTMVAReader;
}

// A function that should work on both pat and reco objects
template <typename T> void ElectronMVAEstimatorRun2Phys14NonTrig::
fillMVAVariables(const T& particle){

  // Both pat and reco particles have exactly the same accessors.
  auto superCluster = particle->superCluster();
  bool validKF= false; 
  reco::TrackRef myTrackRef = particle->closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable() && (myTrackRef.isNonnull()) );  
	     
  _allMVAVars.kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  // Pure ECAL -> shower shapes
  _allMVAVars.see            = particle->full5x5_sigmaIEtaIEta();;
  _allMVAVars.spp            = particle->full5x5_sigmaIPhiIPhi();
  _allMVAVars.OneMinusE1x5E5x5 = 1. - particle->full5x5_e1x5() / particle->full5x5_e5x5();
  _allMVAVars.R9             = particle->full5x5_r9();
  _allMVAVars.etawidth       = superCluster->etaWidth();
  _allMVAVars.phiwidth       = superCluster->phiWidth();
  _allMVAVars.HoE            = particle->hadronicOverEm();
  // Endcap only variables
  _allMVAVars.PreShowerOverRaw  = superCluster->preshowerEnergy() / superCluster->rawEnergy();
  //Pure tracking variables
  _allMVAVars.kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  _allMVAVars.gsfchi2         = particle->gsfTrack()->normalizedChi2();
  // Energy matching
  _allMVAVars.fbrem           = particle->fbrem();
  _allMVAVars.EoP             = particle->eSuperClusterOverP();
  _allMVAVars.eleEoPout       = particle->eEleClusterOverPout();
  _allMVAVars.IoEmIoP         = (1.0/particle->ecalEnergy()) - (1.0 / particle->p());
  // Geometrical matchings
  _allMVAVars.deta            = particle->deltaEtaSuperClusterTrackAtVtx();
  _allMVAVars.dphi            = particle->deltaPhiSuperClusterTrackAtVtx();
  _allMVAVars.detacalo        = particle->deltaEtaSeedClusterTrackAtCalo();
  // Spectator variables  
  _allMVAVars.pt              = particle->pt();
  float scEta = superCluster->eta();
  _allMVAVars.isBarrel        = ( fabs(scEta) < 1.479 );
  _allMVAVars.isEndcap        = ( fabs(scEta) >= 1.479);
  _allMVAVars.SCeta           = scEta;


}


