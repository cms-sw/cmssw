#include <TFile.h>
#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimatorCSA14.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <zlib.h>
#include "TMVA/MethodBase.h"


//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimatorCSA14::EGammaMvaEleEstimatorCSA14() :
fMethodname("BDTG method"),
fisInitialized(kFALSE),
fMVAType(kTrig),
fUseBinnedVersion(kTRUE),
fNMVABins(0)
{
  // Constructor.  
}

//--------------------------------------------------------------------------------------------------
EGammaMvaEleEstimatorCSA14::~EGammaMvaEleEstimatorCSA14()
{
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAMethod[i]) delete fTMVAMethod[i];
    if (fTMVAReader[i]) delete fTMVAReader[i];
  }
}

//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimatorCSA14::initialize( std::string methodName,
                                       std::string weightsfile,
                                       EGammaMvaEleEstimatorCSA14::MVAType type)
{
  
  std::vector<std::string> tempWeightFileVector;
  tempWeightFileVector.push_back(weightsfile);
  initialize(methodName,type,kFALSE,tempWeightFileVector);
}


//--------------------------------------------------------------------------------------------------
void EGammaMvaEleEstimatorCSA14::initialize( std::string methodName,
                                       EGammaMvaEleEstimatorCSA14::MVAType type,
                                       Bool_t useBinnedVersion,
				       std::vector<std::string> weightsfiles
  ) {

  //clean up first
  for (unsigned int i=0;i<fTMVAReader.size(); ++i) {
    if (fTMVAReader[i]) delete fTMVAReader[i];
    if (fTMVAMethod[i]) delete fTMVAMethod[i];
  }
  fTMVAReader.clear();
  fTMVAMethod.clear();
  //initialize
  fisInitialized = kTRUE;
  fMVAType = type;
  fMethodname = methodName;
  fUseBinnedVersion = useBinnedVersion;

  //Define expected number of bins
  UInt_t ExpectedNBins = 0;
  if (type == kTrig) {
    ExpectedNBins = 2;
  }
   else if (type == kNonTrig) {
    ExpectedNBins = 4;
  }
   else if (type ==kNonTrigPhys14) {
    ExpectedNBins = 6;
  }

  fNMVABins = ExpectedNBins;
  
  //Check number of weight files given
  if (fNMVABins != weightsfiles.size() ) {
    std::cout << "Error: Expected Number of bins = " << fNMVABins << " does not equal to weightsfiles.size() = " 
              << weightsfiles.size() << std::endl; 
 
   #ifndef STANDALONE
    assert(fNMVABins == weightsfiles.size());
   #endif 
  }

  //Loop over all bins
  for (unsigned int i=0;i<fNMVABins; ++i) {
  
    TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:!Silent:Error" );  
    tmpTMVAReader->SetVerbose(kTRUE);
  
    if (type == kTrig) {
      // Pure tracking variables
        // Pure tracking variables
        tmpTMVAReader->AddVariable("fBrem",           &fMVAVar_fbrem);
        tmpTMVAReader->AddVariable("kfchi2",          &fMVAVar_kfchi2);
        tmpTMVAReader->AddVariable("kfhits",          &fMVAVar_kfhits);
        tmpTMVAReader->AddVariable("gsfChi2",         &fMVAVar_gsfchi2);
        
        // Geometrical matchings
        tmpTMVAReader->AddVariable("eledeta",            &fMVAVar_deta);
        tmpTMVAReader->AddVariable("eledphi",            &fMVAVar_dphi);
        tmpTMVAReader->AddVariable("detacalo",        &fMVAVar_detacalo);
        
        // Pure ECAL -> shower shapes
        tmpTMVAReader->AddVariable("noZSsee",             &fMVAVar_see);
        tmpTMVAReader->AddVariable("noZSspp",             &fMVAVar_spp);
        tmpTMVAReader->AddVariable("etawidth",        &fMVAVar_etawidth);
        tmpTMVAReader->AddVariable("phiwidth",        &fMVAVar_phiwidth);
        tmpTMVAReader->AddVariable("noZSe1x5e5x5",        &fMVAVar_OneMinusE1x5E5x5);
        tmpTMVAReader->AddVariable("noZSr9",              &fMVAVar_R9);
        
        // Energy matching
        tmpTMVAReader->AddVariable("HtoE",             &fMVAVar_HoE);
        tmpTMVAReader->AddVariable("EoP",             &fMVAVar_EoP);
        tmpTMVAReader->AddVariable("IoEmIoP",         &fMVAVar_IoEmIoP);
        tmpTMVAReader->AddVariable("EEleoPout",       &fMVAVar_eleEoPout);
        if(i == 1 ) tmpTMVAReader->AddVariable("PreShowerOverRaw",&fMVAVar_PreShowerOverRaw);

        tmpTMVAReader->AddSpectator("pt",             &fMVAVar_pt);
        tmpTMVAReader->AddSpectator("absEta",            &fMVAVar_abseta);
    }
  
  
    if ((type == kNonTrig)||(type == kNonTrigPhys14)) {
        
        tmpTMVAReader->AddVariable("ele_kfhits",          &fMVAVar_kfhits);
        // Pure ECAL -> shower shapes
        tmpTMVAReader->AddVariable("ele_oldsigmaietaieta",             &fMVAVar_see);
        tmpTMVAReader->AddVariable("ele_oldsigmaiphiiphi",             &fMVAVar_spp);
        tmpTMVAReader->AddVariable("ele_oldcircularity",        &fMVAVar_OneMinusE1x5E5x5);
        tmpTMVAReader->AddVariable("ele_oldr9",              &fMVAVar_R9);
        tmpTMVAReader->AddVariable("ele_scletawidth",        &fMVAVar_etawidth);
        tmpTMVAReader->AddVariable("ele_sclphiwidth",        &fMVAVar_phiwidth);
        tmpTMVAReader->AddVariable("ele_he",             &fMVAVar_HoE);
        if ((type == kNonTrig)&&(i == 1 || i == 3)) tmpTMVAReader->AddVariable("ele_psEoverEraw",&fMVAVar_PreShowerOverRaw);
        if ((type == kNonTrigPhys14)&&(i == 2 || i == 5)) tmpTMVAReader->AddVariable("ele_psEoverEraw",&fMVAVar_PreShowerOverRaw);
        
        
        //Pure tracking variables
        tmpTMVAReader->AddVariable("ele_kfchi2",          &fMVAVar_kfchi2);
        tmpTMVAReader->AddVariable("ele_chi2_hits",         &fMVAVar_gsfchi2);
        // Energy matching
        tmpTMVAReader->AddVariable("ele_fbrem",           &fMVAVar_fbrem);
        tmpTMVAReader->AddVariable("ele_ep",             &fMVAVar_EoP);
        tmpTMVAReader->AddVariable("ele_eelepout",       &fMVAVar_eleEoPout);
        tmpTMVAReader->AddVariable("ele_IoEmIop",         &fMVAVar_IoEmIoP);
        
        // Geometrical matchings
        tmpTMVAReader->AddVariable("ele_deltaetain",            &fMVAVar_deta);
        tmpTMVAReader->AddVariable("ele_deltaphiin",            &fMVAVar_dphi);
        tmpTMVAReader->AddVariable("ele_deltaetaseed",        &fMVAVar_detacalo);
    


        tmpTMVAReader->AddSpectator("ele_pT",             &fMVAVar_pt);
        tmpTMVAReader->AddSpectator("ele_isbarrel",             &fMVAVar_isBarrel);
        tmpTMVAReader->AddSpectator("ele_isendcap",             &fMVAVar_isEndcap);
        if (type == kNonTrigPhys14) tmpTMVAReader->AddSpectator("scl_eta",                  &fMVAVar_SCeta);

    }

 


#ifndef STANDALONE
    if ((fMethodname.find("BDT") == 0) && (weightsfiles[i].rfind(".xml.gz") == weightsfiles[i].length()-strlen(".xml.gz"))) {
        gzFile file = gzopen(weightsfiles[i].c_str(), "rb");
        if (file == nullptr) { std::cout  << "Error opening gzip file associated to " << weightsfiles[i] << std::endl; throw cms::Exception("Configuration","Error reading zipped XML file"); }
        std::vector<char> data; 
        data.reserve(1024*1024*10);
        unsigned int bufflen = 32*1024;
        char *buff = reinterpret_cast<char *>(malloc(bufflen));
        if (buff == nullptr) { std::cout  << "Error creating buffer for " << weightsfiles[i] << std::endl;  gzclose(file); throw cms::Exception("Configuration","Error reading zipped XML file"); }
        int read;
        while ((read = gzread(file, buff, bufflen)) != 0) {
            if (read == -1) { std::cout  << "Error reading gzip file associated to " << weightsfiles[i] << ": " << gzerror(file,&read) << std::endl; gzclose(file); free(buff); throw cms::Exception("Configuration","Error reading zipped XML file"); }
            data.insert(data.end(), buff, buff+read);
        }
        if (gzclose(file) != Z_OK) { std::cout  << "Error closing gzip file associated to " << weightsfiles[i] << std::endl; }
        free(buff);
        data.push_back('\0'); // IMPORTANT
        fTMVAMethod.push_back(dynamic_cast<TMVA::MethodBase*>(tmpTMVAReader->BookMVA(TMVA::Types::kBDT, &data[0])));
    } else {
        if (weightsfiles[i].rfind(".xml.gz") == weightsfiles[i].length()-strlen(".xml.gz")) {
            std::cout  << "Error: xml.gz unsupported for method " << fMethodname << ", weight file " << weightsfiles[i] << std::endl; throw cms::Exception("Configuration","Error reading zipped XML file"); 
        }
        fTMVAMethod.push_back(dynamic_cast<TMVA::MethodBase*>(tmpTMVAReader->BookMVA(fMethodname , weightsfiles[i])));
    }
#else
    if (weightsfiles[i].rfind(".xml.gz") == weightsfiles[i].length()-strlen(".xml.gz")) {
        std::cout  << "Error: xml.gz unsupported for method " << fMethodname << ", weight file " << weightsfiles[i] << std::endl; abort();
    }
    fTMVAMethod.push_back(dynamic_cast<TMVA::MethodBase*>(tmpTMVAReader->BookMVA(fMethodname , weightsfiles[i])));
#endif
    std::cout << "MVABin " << i << " : MethodName = " << fMethodname 
              << " , type == " << type << " , "
              << "Load weights file : " << weightsfiles[i] 
              << std::endl;
    fTMVAReader.push_back(tmpTMVAReader);
  }
  std::cout << "Electron ID MVA Completed\n";

}


//--------------------------------------------------------------------------------------------------
UInt_t EGammaMvaEleEstimatorCSA14::GetMVABin( double eta, double pt) const {
  
    //Default is to return the first bin
    unsigned int bin = 0;


    if (fMVAType == EGammaMvaEleEstimatorCSA14::kNonTrig ) {
      bin = 0;
      if (pt < 10 && fabs(eta) < 1.479) bin = 0;
      if (pt < 10 && fabs(eta) >= 1.479) bin = 1;
      if (pt >= 10 && fabs(eta) < 1.479) bin = 2;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 3;
    }


    if (fMVAType == EGammaMvaEleEstimatorCSA14::kTrig
      ) {
      bin = 0;
      if (pt >= 10 && fabs(eta) < 1.479) bin = 0;
      if (pt >= 10 && fabs(eta) >= 1.479) bin = 1;
    }

    if (fMVAType == EGammaMvaEleEstimatorCSA14::kNonTrigPhys14 ){
       bin = 0;
       if (pt < 10 && fabs(eta) < 0.8) bin = 0;
       if (pt < 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479) bin = 1;
       if (pt < 10 && fabs(eta) >= 1.479) bin = 2;
       if (pt >= 10 && fabs(eta) < 0.8) bin = 3;
       if (pt >= 10 && fabs(eta) >= 0.8 && fabs(eta) < 1.479) bin = 4;
       if (pt >= 10 && fabs(eta) >= 1.479) bin = 5;
    }

    return bin;
}





//--------------------------------------------------------------------------------------------------

// for kTrig and kNonTrig algorithm
Double_t EGammaMvaEleEstimatorCSA14::mvaValue(const reco::GsfElectron& ele, 
					const reco::Vertex& vertex, 
					const TransientTrackBuilder& transientTrackBuilder,					
                                              noZS::EcalClusterLazyTools myEcalCluster,
					bool printDebug) {
  
  if (!fisInitialized) { 
    std::cout << "Error: EGammaMvaEleEstimatorCSA14 not properly initialized.\n"; 
    return -9999;
  }

  if ( (fMVAType != EGammaMvaEleEstimatorCSA14::kTrig) && (fMVAType != EGammaMvaEleEstimatorCSA14::kNonTrig) && (fMVAType != EGammaMvaEleEstimatorCSA14::kNonTrigPhys14) ) {
    std::cout << "Error: This method should be called for kTrig or kNonTrig or kNonTrigPhys14 MVA only" << endl;
    return -9999;
  }
 
  bool validKF= false; 
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());  

  // Pure tracking variables
  fMVAVar_fbrem           =  ele.fbrem();
  fMVAVar_kfchi2          =  (validKF) ? myTrackRef->normalizedChi2() : 0 ;
  fMVAVar_kfhits          =  (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ; 
  fMVAVar_kfhitsall          =  (validKF) ? myTrackRef->numberOfValidHits() : -1. ;   //  save also this in your ntuple as possible alternative
  fMVAVar_gsfchi2         =  ele.gsfTrack()->normalizedChi2();  

  
  // Geometrical matchings
  fMVAVar_deta            =  ele.deltaEtaSuperClusterTrackAtVtx();
  fMVAVar_dphi            =  ele.deltaPhiSuperClusterTrackAtVtx();
  fMVAVar_detacalo        =  ele.deltaEtaSeedClusterTrackAtCalo();


  // Pure ECAL -> shower shapes
  std::vector<float> vCov = myEcalCluster.localCovariances(*(ele.superCluster()->seed())) ;
  if (!isnan(vCov[0])) fMVAVar_see = sqrt (vCov[0]); //EleSigmaIEtaIEta
  else fMVAVar_see = 0.;
  if (!isnan(vCov[2])) fMVAVar_spp = sqrt (vCov[2]);   //EleSigmaIPhiIPhi
  else fMVAVar_spp = 0.;    

  fMVAVar_etawidth        =  ele.superCluster()->etaWidth();
  fMVAVar_phiwidth        =  ele.superCluster()->phiWidth();
  fMVAVar_OneMinusE1x5E5x5        =  (ele.e5x5()) !=0. ? 1.-(myEcalCluster.e1x5(*(ele.superCluster()->seed()))/myEcalCluster.e5x5(*(ele.superCluster()->seed()))) : -1. ;
  fMVAVar_R9              =  myEcalCluster.e3x3(*(ele.superCluster()->seed())) / ele.superCluster()->rawEnergy();

  // Energy matching
  fMVAVar_HoE             =  ele.hadronicOverEm();
  fMVAVar_EoP             =  ele.eSuperClusterOverP();
  fMVAVar_IoEmIoP         =  (1.0/ele.ecalEnergy()) - (1.0 / ele.p());  // in the future to be changed with ele.gsfTrack()->p()
  fMVAVar_eleEoPout       =  ele.eEleClusterOverPout();
  fMVAVar_PreShowerOverRaw=  ele.superCluster()->preshowerEnergy() / ele.superCluster()->rawEnergy();


  // Spectators
  fMVAVar_eta             =  ele.superCluster()->eta();
  fMVAVar_abseta          =  fabs(ele.superCluster()->eta());
  fMVAVar_pt              =  ele.pt();                          
  fMVAVar_isBarrel        =  (ele.superCluster()->eta()<1.479);
  fMVAVar_isEndcap        =  (ele.superCluster()->eta()>1.479);
  fMVAVar_SCeta           =  ele.superCluster()->eta();
 

  // for triggering electrons get the impact parameteres
  if(fMVAType == kTrig) {
    //d0
    if (ele.gsfTrack().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.gsfTrack()->dxy(vertex.position()); 
    } else if (ele.closestCtfTrackRef().isNonnull()) {
      fMVAVar_d0 = (-1.0)*ele.closestCtfTrackRef()->dxy(vertex.position()); 
    } else {
      fMVAVar_d0 = -9999.0;
    }
    
    //default values for IP3D
    fMVAVar_ip3d = -999.0; 
    fMVAVar_ip3dSig = 0.0;
    if (ele.gsfTrack().isNonnull()) {
      const double gsfsign   = ( (-ele.gsfTrack()->dxy(vertex.position()))   >=0 ) ? 1. : -1.;
      
      const reco::TransientTrack &tt = transientTrackBuilder.build(ele.gsfTrack()); 
      const std::pair<bool,Measurement1D> &ip3dpv =  IPTools::absoluteImpactParameter3D(tt,vertex);
      if (ip3dpv.first) {
	double ip3d = gsfsign*ip3dpv.second.value();
	double ip3derr = ip3dpv.second.error();  
	fMVAVar_ip3d = ip3d; 
        fMVAVar_ip3dSig = ip3d/ip3derr;
      }
    }
  }
  
  // evaluate
  bindVariables();
  Double_t mva = -9999;  
  if (fUseBinnedVersion) {
    int bin = GetMVABin(fMVAVar_eta,fMVAVar_pt);
    mva = fTMVAReader[bin]->EvaluateMVA(fTMVAMethod[bin]);
  } else {
    mva = fTMVAReader[0]->EvaluateMVA(fTMVAMethod[0]);
  }



  if(printDebug) {
    cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
    cout << " fbrem " <<  fMVAVar_fbrem  
      	 << " kfchi2 " << fMVAVar_kfchi2  
	 << " mykfhits " << fMVAVar_kfhits  
	 << " gsfchi2 " << fMVAVar_gsfchi2  
	 << " deta " <<  fMVAVar_deta  
	 << " dphi " << fMVAVar_dphi  
      	 << " detacalo " << fMVAVar_detacalo  
	 << " see " << fMVAVar_see  
	 << " spp " << fMVAVar_spp  
	 << " etawidth " << fMVAVar_etawidth  
	 << " phiwidth " << fMVAVar_phiwidth  
	 << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
	 << " R9 " << fMVAVar_R9  
	 << " HoE " << fMVAVar_HoE  
	 << " EoP " << fMVAVar_EoP  
	 << " IoEmIoP " << fMVAVar_IoEmIoP  
	 << " eleEoPout " << fMVAVar_eleEoPout  
	 << " d0 " << fMVAVar_d0  
	 << " ip3d " << fMVAVar_ip3d  
	 << " eta " << fMVAVar_eta  
	 << " pt " << fMVAVar_pt << endl;
    cout << " ### MVA " << mva << endl;
  }



  return mva;
}



Double_t EGammaMvaEleEstimatorCSA14::mvaValue(const pat::Electron& ele,
                                              bool printDebug) {
    
    if (!fisInitialized) {
        std::cout << "Error: EGammaMvaEleEstimatorCSA14 not properly initialized.\n";
        return -9999;
    }
    
    if ( (fMVAType != EGammaMvaEleEstimatorCSA14::kTrig) && (fMVAType != EGammaMvaEleEstimatorCSA14::kNonTrig) && (fMVAType != EGammaMvaEleEstimatorCSA14::kNonTrigPhys14) ) {
        std::cout << "Error: This method should be called for kTrig or kNonTrig or kNonTrigPhys14 MVA only" << endl;
        return -9999;
    }
    
    bool validKF= false;
    reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
    validKF = (myTrackRef.isAvailable());
    validKF = (myTrackRef.isNonnull());
    
    // Pure tracking variables
    fMVAVar_fbrem           =  ele.fbrem();
    fMVAVar_kfchi2          =  (validKF) ? myTrackRef->normalizedChi2() : 0 ;
    fMVAVar_kfhits          =  (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
    fMVAVar_kfhitsall          =  (validKF) ? myTrackRef->numberOfValidHits() : -1. ;   //  save also this in your ntuple as possible alternative
    fMVAVar_gsfchi2         =  ele.gsfTrack()->normalizedChi2();
    
    
    // Geometrical matchings
    fMVAVar_deta            =  ele.deltaEtaSuperClusterTrackAtVtx();
    fMVAVar_dphi            =  ele.deltaPhiSuperClusterTrackAtVtx();
    fMVAVar_detacalo        =  ele.deltaEtaSeedClusterTrackAtCalo();
    
    
    // Pure ECAL -> shower shapes
    fMVAVar_see = ele.full5x5_sigmaIetaIeta(); //EleSigmaIEtaIEta
    fMVAVar_spp = ele.full5x5_sigmaIphiIphi();   //EleSigmaIPhiIPhi
    
    fMVAVar_etawidth        =  ele.superCluster()->etaWidth();
    fMVAVar_phiwidth        =  ele.superCluster()->phiWidth();
    fMVAVar_OneMinusE1x5E5x5        =  (ele.full5x5_e5x5()) !=0. ? 1.-(ele.full5x5_e1x5()/ele.full5x5_e5x5()) : -1. ;
    fMVAVar_R9              =  ele.full5x5_r9();
    
    // Energy matching
    fMVAVar_HoE             =  ele.hadronicOverEm();
    fMVAVar_EoP             =  ele.eSuperClusterOverP();
    fMVAVar_IoEmIoP         =  (1.0/ele.ecalEnergy()) - (1.0 / ele.p());  // in the future to be changed with ele.gsfTrack()->p()
    fMVAVar_eleEoPout       =  ele.eEleClusterOverPout();
    fMVAVar_PreShowerOverRaw=  ele.superCluster()->preshowerEnergy() / ele.superCluster()->rawEnergy();
    
    
    // Spectators
    fMVAVar_eta             =  ele.superCluster()->eta();
    fMVAVar_abseta          =  fabs(ele.superCluster()->eta());
    fMVAVar_pt              =  ele.pt();
    fMVAVar_isBarrel        =  (ele.superCluster()->eta()<1.479);
    fMVAVar_isEndcap        =  (ele.superCluster()->eta()>1.479);
    fMVAVar_SCeta           =  ele.superCluster()->eta();
    

    
    // evaluate
    bindVariables();
    Double_t mva = -9999;
    if (fUseBinnedVersion) {
        int bin = GetMVABin(fMVAVar_eta,fMVAVar_pt);
        mva = fTMVAReader[bin]->EvaluateMVA(fTMVAMethod[bin]);
    } else {
        mva = fTMVAReader[0]->EvaluateMVA(fTMVAMethod[0]);
    }
    
    
    
    if(printDebug) {
        cout << " *** Inside the class fMethodname " << fMethodname << " fMVAType " << fMVAType << endl;
        cout << " fbrem " <<  fMVAVar_fbrem
        << " kfchi2 " << fMVAVar_kfchi2
        << " mykfhits " << fMVAVar_kfhits
        << " gsfchi2 " << fMVAVar_gsfchi2
        << " deta " <<  fMVAVar_deta
        << " dphi " << fMVAVar_dphi
        << " detacalo " << fMVAVar_detacalo
        << " see " << fMVAVar_see
        << " spp " << fMVAVar_spp
        << " etawidth " << fMVAVar_etawidth  
        << " phiwidth " << fMVAVar_phiwidth  
        << " OneMinusE1x5E5x5 " << fMVAVar_OneMinusE1x5E5x5  
        << " R9 " << fMVAVar_R9  
        << " HoE " << fMVAVar_HoE  
        << " EoP " << fMVAVar_EoP  
        << " IoEmIoP " << fMVAVar_IoEmIoP  
        << " eleEoPout " << fMVAVar_eleEoPout  
        << " eta " << fMVAVar_eta
        << " pt " << fMVAVar_pt << endl;
        cout << " ### MVA " << mva << endl;
    }
    
    
    
    return mva;
}



void EGammaMvaEleEstimatorCSA14::bindVariables() {

  // this binding is needed for variables that sometime diverge. 


  if(fMVAVar_fbrem < -1.)
    fMVAVar_fbrem = -1.;	
  
  fMVAVar_deta = fabs(fMVAVar_deta);
  if(fMVAVar_deta > 0.06)
    fMVAVar_deta = 0.06;
  
  
  fMVAVar_dphi = fabs(fMVAVar_dphi);
  if(fMVAVar_dphi > 0.6)
    fMVAVar_dphi = 0.6;
  

  if(fMVAVar_EoP > 20.)
    fMVAVar_EoP = 20.;
  
  if(fMVAVar_eleEoPout > 20.)
    fMVAVar_eleEoPout = 20.;
  
  
  fMVAVar_detacalo = fabs(fMVAVar_detacalo);
  if(fMVAVar_detacalo > 0.2)
    fMVAVar_detacalo = 0.2;
  
  if(fMVAVar_OneMinusE1x5E5x5 < -1.)
    fMVAVar_OneMinusE1x5E5x5 = -1;
  
  if(fMVAVar_OneMinusE1x5E5x5 > 2.)
    fMVAVar_OneMinusE1x5E5x5 = 2.; 
  
  
  
  if(fMVAVar_R9 > 5)
    fMVAVar_R9 = 5;
  
  if(fMVAVar_gsfchi2 > 200.)
    fMVAVar_gsfchi2 = 200;
  
  
  if(fMVAVar_kfchi2 > 10.)
    fMVAVar_kfchi2 = 10.;
  
  
  // Needed for a bug in CMSSW_420, fixed in more recent CMSSW versions
  if(std::isnan(fMVAVar_spp))
    fMVAVar_spp = 0.;	
  
  
  return;
}








