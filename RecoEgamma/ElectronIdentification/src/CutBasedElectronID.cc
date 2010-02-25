#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
//#include "DataFormats/TrackReco/interface/Track.h"

void CutBasedElectronID::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);
  
  type_ = conf.getParameter<std::string>("electronIDType");
  quality_ = conf.getParameter<std::string>("electronQuality");
  version_ = conf.getParameter<std::string>("electronVersion");
  verticesCollection = conf.getParameter<edm::InputTag>("verticesCollection");
  
  if (type_ == "robust" || type_ == "classbased") {
    if (quality_ == "loose" || quality_ == "tight" ||
        quality_ == "medium" || quality_ == "highenergy" ) {
       std::string stringCut = type_+quality_+"EleIDCuts"+version_;
       cuts_ = conf.getParameter<edm::ParameterSet>(stringCut);
    }
    else {
       edm::LogError("CutBasedElectronID") << "Invalid electronQuality parameter: must be loose, tight or highenergy." ;
       exit (1);
    }
  } 
  else {
    edm::LogError("CutBasedElectronID") << "Invalid electronType parameter: must be robust or classbased." ;
    exit (1);
  }
}

int CutBasedElectronID::classify(const reco::GsfElectron* electron) {
  
  double eOverP = electron->eSuperClusterOverP();
  //double pin  = electron->trackMomentumAtVtx().R(); 
  //double pout = electron->trackMomentumOut().R(); 
  //double fBrem = (pin-pout)/pin;
  double fBrem = electron->fbrem();

  int cat = -1;
  if (version_ == "V00" || version_ == "V01") {
    if((electron->isEB() && fBrem<0.06) || (electron->isEE() && fBrem<0.1)) 
      cat=1;
    else if (eOverP < 1.2 && eOverP > 0.8) 
      cat=0;
    else 
      cat=2;
    
    return cat;

  } else {
    if (electron->isEB()) {       // BARREL
      if(fBrem < 0.12)
        cat=1;
      else if (eOverP < 1.2 && eOverP > 0.9) 
        cat=0;
      else 
        cat=2;
    } else {                     // ENDCAP
      if(fBrem < 0.2)
        cat=1;
      else if (eOverP < 1.22 && eOverP > 0.82) 
        cat=0;
      else 
        cat=2;
    }
    
    return cat;
  }
}

double CutBasedElectronID::result(const reco::GsfElectron* electron ,
                                  const edm::Event& e ,
                                  const edm::EventSetup& es) { 
  
  double eta = electron->p4().Eta();
  double eOverP = electron->eSuperClusterOverP();
  double eSeed = electron->superCluster()->seed()->energy();
  double pin  = electron->trackMomentumAtVtx().R();   
  double pout = electron->trackMomentumOut().R(); 
  double eSeedOverPin = eSeed/pin; 
  double fBrem = (pin-pout)/pin;
  double hOverE = electron->hadronicOverEm();
  EcalClusterLazyTools lazyTools = getClusterShape(e,es);
  std::vector<float> vLocCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  double sigmaee = sqrt(vLocCov[0]);
  double e25Max = lazyTools.e2x5Max(*(electron->superCluster()->seed()))  ;
  double e15 = lazyTools.e1x5(*(electron->superCluster()->seed()))  ;
  double e55 = lazyTools.e5x5(*(electron->superCluster()->seed())) ;
  double e25Maxoe55 = e25Max/e55 ;
  double e15oe55 = e15/e55 ;
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = 0;
  int mishits = electron->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
  double tkIso = electron->dr03TkSumPt();
  double ecalIso = electron->dr04EcalRecHitSumEt();
  double hcalIso = electron->dr04HcalTowerSumEt();

  if (version_ == "V00") {
     std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;
     sigmaee = sqrt(vCov[0]);  
     if (electron->isEE())
       sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  if (version_ == "V02" || version_ == "") {
    edm::Handle<reco::VertexCollection> vtxH;
    e.getByLabel(verticesCollection, vtxH);
    if (vtxH->size() != 0) {
      reco::VertexRef vtx(vtxH, 0);
      ip = fabs(electron->gsfTrack()->dxy(math::XYZPoint(vtx->x(),vtx->y(),vtx->z())));
    } else
      ip = fabs(electron->gsfTrack()->dxy());
    
    if (electron->isEB()) {
      std::vector<float> vCov = lazyTools.scLocalCovariances(*(electron->superCluster()));
      sigmaee = sqrt(vCov[0]); 
    } 
  }
  
  std::vector<double> cut;
  // ROBUST Selection
  if (type_ == "robust") {
    
    float result = 0;

    // hoe, sigmaEtaEta, dPhiIn, dEtaIn
    if (electron->isEB())
      cut = cuts_.getParameter<std::vector<double> >("barrel");
    else
      cut = cuts_.getParameter<std::vector<double> >("endcap");

    if ((tkIso > cut[6]) || (ecalIso > cut[7]) || (hcalIso > cut[8]))
      result = 0.;
    else
      result = 2.;

    if (hOverE > cut[0]) 
      return result;    

    if (sigmaee > cut[1]) 
      return result;    

    if (fabs(deltaPhiIn) > cut[2]) 
      return result;    

    if (fabs(deltaEtaIn) > cut[3]) 
      return result;    
    
    if (e25Maxoe55 < cut[4] && e15oe55 < cut[5])
         return result;
    
    result = result + 1;
    
    return 1.;
  }
  
  int cat = classify(electron);
  int eb;

  if (electron->isEB()) 
    eb = 0;
  else 
    eb = 1; 

  // LOOSE and TIGHT Selections
  if (type_ == "classbased" && (version_ == "V01" || version_ == "V00")) {
    
    if ((eOverP < 0.8) && (fBrem < 0.2)) 
      return 0.;
    
    cut = cuts_.getParameter<std::vector<double> >("hOverE");
    if (hOverE > cut[cat+4*eb]) 
      return 0.;    
    
    cut = cuts_.getParameter<std::vector<double> >("sigmaEtaEta");
    if (sigmaee > cut[cat+4*eb]) 
      return 0.;    
    
    cut = cuts_.getParameter<std::vector<double> >("deltaPhiIn");
    if (eOverP < 1.5) {
      if (fabs(deltaPhiIn) > cut[cat+4*eb]) 
        return 0.;    
    } else {
      if (fabs(deltaPhiIn) > cut[3+4*eb])
        return 0.;
    }
    
    cut = cuts_.getParameter<std::vector<double> >("deltaEtaIn");
    if (fabs(deltaEtaIn) > cut[cat+4*eb]) 
      return 0.;    
    
    cut = cuts_.getParameter<std::vector<double> >("eSeedOverPin");
    if (eSeedOverPin < cut[cat+4*eb]) 
      return 0.;  
    
    if (quality_ == "tight")
      if (eOverP < 0.9*(1-fBrem))
        return 0.;
    
    return 1.;
  }
  
  if (type_ == "classbased" && (version_ == "V02" || version_ == "")) {
    double result = 0.;
    
    double scTheta = (2*atan(exp(-electron->superCluster()->eta())));
    double scEt = electron->superCluster()->energy()*sin(scTheta);
  
    int bin = 0;

    if (scEt < 20.)
      bin = 2;
    else if (scEt > 30.)
      bin = 0;
    else
      bin = 1;

    if (fBrem > 0)
      eSeedOverPin = eSeedOverPin + fBrem;
    
    if (bin != 2) {     
      tkIso = tkIso*pow(40./scEt, 2); 
      ecalIso = ecalIso*pow(40./scEt, 2); 
      hcalIso = hcalIso*pow(40./scEt, 2); 
    }

    std::vector<double> cutTk = cuts_.getParameter<std::vector<double> >("cutisotk");
    std::vector<double> cutEcal = cuts_.getParameter<std::vector<double> >("cutisoecal");
    std::vector<double> cutHcal = cuts_.getParameter<std::vector<double> >("cutisohcal");
    if ((tkIso > cutTk[cat+3*eb+bin*6]) ||
        (ecalIso > cutEcal[cat+3*eb+bin*6]) ||
        (hcalIso > cutHcal[cat+3*eb+bin*6]))
      result = 0.;
    else
      result = 2.;

    if (fBrem < -2)
      return result;

    //std::cout << "hoe" << hOverE << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cuthoe");
    if (hOverE > cut[cat+3*eb+bin*6])
      return result;

    //std::cout << "see" << sigmaee << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cutsee");
    if (sigmaee > cut[cat+3*eb+bin*6])
      return result;

    //std::cout << "dphiin" << fabs(deltaPhiIn) << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cutdphi");
    if (fabs(deltaPhiIn) > cut[cat+3*eb+bin*6])
      return result;  

    //std::cout << "detain" << fabs(deltaEtaIn) << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cutdeta");
    if (fabs(deltaEtaIn) > cut[cat+3*eb+bin*6])
      return result;

    //std::cout << "eseedopin " << eSeedOverPin << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cuteopin");
    if (eSeedOverPin < cut[cat+3*eb+bin*6])
      return result;

    //std::cout << "ip" << ip << std::endl;
    cut = cuts_.getParameter<std::vector<double> >("cutip");
    if (ip > cut[cat+3*eb+bin*6])
      return result;

    cut = cuts_.getParameter<std::vector<double> >("cutmishits");
    if (mishits > cut[cat+3*eb+bin*6])
      return result;

    result = result + 1.;

    return result;
  }

  return -1.;
}
