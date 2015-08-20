#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Phys14NonTrig.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

ElectronMVAEstimatorRun2Phys14NonTrig::ElectronMVAEstimatorRun2Phys14NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf) {

  _tag = conf.getParameter<std::string>("mvaTag");
  
  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  if( (int)(weightFileNames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _gbrForests.clear();
  _MethodName = "BDTG method";
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    _gbrForests.push_back( createSingleReader(i, weightFile ) );

  }

}

ElectronMVAEstimatorRun2Phys14NonTrig::
~ElectronMVAEstimatorRun2Phys14NonTrig(){
}

float ElectronMVAEstimatorRun2Phys14NonTrig::
mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event& evt) const {
  
  const int iCategory = findCategory( particle );
  const std::vector<float> vars = std::move( fillMVAVariables( particle, evt ) );  
  const float result = _gbrForests.at(iCategory)->GetClassifier(vars.data());

  constexpr bool debug = false;
  if(debug) {
    std::cout << " *** Inside the class _MethodName " << _MethodName << std::endl;
    std::cout << " bin " << iCategory
	      << " fbrem "    << vars[11]  //_allMVAVars.fbrem
              << " kfchi2 "   << vars[9]  //_allMVAVars.kfchi2
              << " mykfhits " << vars[0]  //_allMVAVars.kfhits
              << " gsfchi2 "  << vars[10]  //_allMVAVars.gsfchi2
              << " deta "     << vars[15]  //_allMVAVars.deta
              << " dphi "     << vars[16]  //_allMVAVars.dphi
              << " detacalo " << vars[17]  //_allMVAVars.detacalo
              << " see "      << vars[1]  //_allMVAVars.see
              << " spp "      << vars[2]  //_allMVAVars.spp
              << " etawidth " << vars[5]  //_allMVAVars.etawidth
              << " phiwidth " << vars[6] // _allMVAVars.phiwidth
              << " OneMinusE1x5E5x5 " << vars[3] //_allMVAVars.OneMinusE1x5E5x5
              << " R9 "        << vars[4] //_allMVAVars.R9
              << " HoE "       << vars[7] //_allMVAVars.HoE
              << " EoP "       << vars[12] //_allMVAVars.EoP
              << " IoEmIoP "   << vars[14] //_allMVAVars.IoEmIoP
              << " eleEoPout " << vars[13] // _allMVAVars.eleEoPout
      //<< " d0 " << _allMVAVars.d0
      //   << " ip3d " << _allMVAVars.ip3d
              << " eta "       << vars[21] //_allMVAVars.SCeta
              << " isBarrel "  << vars[19] //_allMVAVars.isBarrel
              << " isEndcap "  << vars[20] //_allMVAVars.isEndcap
	      << " pt "        << vars[18] //_allMVAVars.pt
              << std::endl;
    std::cout << " ### MVA " << result << std::endl;
  }

  return result;
}

int ElectronMVAEstimatorRun2Phys14NonTrig::findCategory(const edm::Ptr<reco::Candidate>& particle) const {
  
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == NULL )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;

  const float pt = eleRecoPtr->pt();
  const float eta = eleRecoPtr->superCluster()->eta();

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;
  constexpr float ptSplit = 10;   // we have above and below 10 GeV categories
  constexpr float ebSplit = 0.800;// barrel is split into two regions
  constexpr float ebeeSplit = 1.479; // division between barrel and endcap

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

bool ElectronMVAEstimatorRun2Phys14NonTrig::
isEndcapCategory(int category ) const {

  bool isEndcap = false;
  if( category == CAT_EE_PT5to10 || category == CAT_EE_PT10plus )
    isEndcap = true;

  return isEndcap;
}


std::unique_ptr<const GBRForest>
ElectronMVAEstimatorRun2Phys14NonTrig::
createSingleReader(const int iCategory, const edm::FileInPath &weightFile) {

  //
  // Create the reader  
  //
  TMVA::Reader tmpTMVAReader( "!Color:Silent:Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //

  tmpTMVAReader.AddVariable("ele_kfhits",           &_allMVAVars.kfhits);
  
  // Pure ECAL -> shower shapes
  tmpTMVAReader.AddVariable("ele_oldsigmaietaieta", &_allMVAVars.see);
  tmpTMVAReader.AddVariable("ele_oldsigmaiphiiphi", &_allMVAVars.spp);
  tmpTMVAReader.AddVariable("ele_oldcircularity",   &_allMVAVars.OneMinusE1x5E5x5);
  tmpTMVAReader.AddVariable("ele_oldr9",            &_allMVAVars.R9);
  tmpTMVAReader.AddVariable("ele_scletawidth",      &_allMVAVars.etawidth);
  tmpTMVAReader.AddVariable("ele_sclphiwidth",      &_allMVAVars.phiwidth);
  tmpTMVAReader.AddVariable("ele_he",               &_allMVAVars.HoE);
  // Endcap only variables
  if( isEndcapCategory(iCategory) )
    tmpTMVAReader.AddVariable("ele_psEoverEraw",    &_allMVAVars.PreShowerOverRaw);
  
  //Pure tracking variables
  tmpTMVAReader.AddVariable("ele_kfchi2",           &_allMVAVars.kfchi2);
  tmpTMVAReader.AddVariable("ele_chi2_hits",        &_allMVAVars.gsfchi2);

  // Energy matching
  tmpTMVAReader.AddVariable("ele_fbrem",           &_allMVAVars.fbrem);
  tmpTMVAReader.AddVariable("ele_ep",              &_allMVAVars.EoP);
  tmpTMVAReader.AddVariable("ele_eelepout",        &_allMVAVars.eleEoPout);
  tmpTMVAReader.AddVariable("ele_IoEmIop",         &_allMVAVars.IoEmIoP);
  
  // Geometrical matchings
  tmpTMVAReader.AddVariable("ele_deltaetain",      &_allMVAVars.deta);
  tmpTMVAReader.AddVariable("ele_deltaphiin",      &_allMVAVars.dphi);
  tmpTMVAReader.AddVariable("ele_deltaetaseed",    &_allMVAVars.detacalo);
  
  // Spectator variables  
  tmpTMVAReader.AddSpectator("ele_pT",             &_allMVAVars.pt);
  tmpTMVAReader.AddSpectator("ele_isbarrel",       &_allMVAVars.isBarrel);
  tmpTMVAReader.AddSpectator("ele_isendcap",       &_allMVAVars.isEndcap);
  tmpTMVAReader.AddSpectator("scl_eta",            &_allMVAVars.SCeta);

  //
  // Book the method and set up the weights file
  //
  std::unique_ptr<TMVA::IMethod> temp( tmpTMVAReader.BookMVA(_MethodName , weightFile.fullPath() ) );
  
  return std::unique_ptr<const GBRForest> ( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmpTMVAReader.FindMVA(_MethodName) ) ) );
}

// A function that should work on both pat and reco objects
std::vector<float> 
ElectronMVAEstimatorRun2Phys14NonTrig::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle,
                                                        const edm::Event& ) const {

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::GsfElectron> eleRecoPtr = ( edm::Ptr<reco::GsfElectron> )particle;
  if( eleRecoPtr.get() == NULL )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::GsfElectron or pat::Electron," << std::endl
      << " but appears to be neither" << std::endl;

  // the instance of the variables that we will manipulate
  AllVariables allMVAVars;

  // Both pat and reco particles have exactly the same accessors, so we use a reco ptr 
  // throughout the code, with a single exception as of this writing, handled separately below.
  auto superCluster = eleRecoPtr->superCluster();

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

  allMVAVars.kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
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
  //Pure tracking variables
  allMVAVars.kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  allMVAVars.gsfchi2         = eleRecoPtr->gsfTrack()->normalizedChi2();
  // Energy matching
  allMVAVars.fbrem           = eleRecoPtr->fbrem();
  allMVAVars.EoP             = eleRecoPtr->eSuperClusterOverP();
  allMVAVars.eleEoPout       = eleRecoPtr->eEleClusterOverPout();
  allMVAVars.IoEmIoP         = (1.0/eleRecoPtr->ecalEnergy()) - (1.0 / eleRecoPtr->p());
  // Geometrical matchings
  allMVAVars.deta            = eleRecoPtr->deltaEtaSuperClusterTrackAtVtx();
  allMVAVars.dphi            = eleRecoPtr->deltaPhiSuperClusterTrackAtVtx();
  allMVAVars.detacalo        = eleRecoPtr->deltaEtaSeedClusterTrackAtCalo();
  // Spectator variables  
  allMVAVars.pt              = eleRecoPtr->pt();
  const float scEta = superCluster->eta();
  allMVAVars.isBarrel        = ( std::abs(scEta) < 1.479 );
  allMVAVars.isEndcap        = ( std::abs(scEta) >= 1.479);
  allMVAVars.SCeta           = scEta;

  constrainMVAVariables(allMVAVars);

  std::vector<float> vars;

  if( isEndcapCategory( findCategory(particle) ) ) {
    vars = std::move( packMVAVariables( allMVAVars.kfhits,
                                        allMVAVars.see,
                                        allMVAVars.spp,
                                        allMVAVars.OneMinusE1x5E5x5,
                                        allMVAVars.R9,
                                        allMVAVars.etawidth,
                                        allMVAVars.phiwidth,
                                        allMVAVars.HoE,
                                        allMVAVars.PreShowerOverRaw,
                                        allMVAVars.kfchi2,
                                        allMVAVars.gsfchi2,
                                        allMVAVars.fbrem,
                                        allMVAVars.EoP,
                                        allMVAVars.eleEoPout,
                                        allMVAVars.IoEmIoP,
                                        allMVAVars.deta,
                                        allMVAVars.dphi,
                                        allMVAVars.detacalo,
                                        allMVAVars.pt,
                                        allMVAVars.isBarrel,
                                        allMVAVars.isEndcap,
                                        allMVAVars.SCeta )
                      );
  } else {
    vars = std::move( packMVAVariables( allMVAVars.kfhits,
                                        allMVAVars.see,
                                        allMVAVars.spp,
                                        allMVAVars.OneMinusE1x5E5x5,
                                        allMVAVars.R9,
                                        allMVAVars.etawidth,
                                        allMVAVars.phiwidth,
                                        allMVAVars.HoE,
                                        allMVAVars.kfchi2,
                                        allMVAVars.gsfchi2,
                                        allMVAVars.fbrem,
                                        allMVAVars.EoP,
                                        allMVAVars.eleEoPout,
                                        allMVAVars.IoEmIoP,
                                        allMVAVars.deta,
                                        allMVAVars.dphi,
                                        allMVAVars.detacalo,
                                        allMVAVars.pt,
                                        allMVAVars.isBarrel,
                                        allMVAVars.isEndcap,
                                        allMVAVars.SCeta )
                      );
  }

  return vars;
}

void ElectronMVAEstimatorRun2Phys14NonTrig::constrainMVAVariables(AllVariables& vars) const {

  // Check that variables do not have crazy values

  if(vars.fbrem < -1.)
    vars.fbrem = -1.;
  
  vars.deta = std::abs(vars.deta);
  if(vars.deta > 0.06)
    vars.deta = 0.06;
  
  vars.dphi = std::abs(vars.dphi);
  if(vars.dphi > 0.6)
    vars.dphi = 0.6;
  
  if(vars.EoP > 20.)
    vars.EoP = 20.;
  
  if(vars.eleEoPout > 20.)
    vars.eleEoPout = 20.;
    
  vars.detacalo = std::abs(vars.detacalo);
  if(vars.detacalo > 0.2)
    vars.detacalo = 0.2;
  
  if(vars.OneMinusE1x5E5x5 < -1.)
    vars.OneMinusE1x5E5x5 = -1;
  
  if(vars.OneMinusE1x5E5x5 > 2.)
    vars.OneMinusE1x5E5x5 = 2.; 
    
  if(vars.R9 > 5)
    vars.R9 = 5;
  
  if(vars.gsfchi2 > 200.)
    vars.gsfchi2 = 200;
    
  if(vars.kfchi2 > 10.)
    vars.kfchi2 = 10.;
}

