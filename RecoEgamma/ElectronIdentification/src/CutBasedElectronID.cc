#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

void CutBasedElectronID::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);
  
  type_ = conf.getParameter<std::string>("electronIDType");
  quality_ = conf.getParameter<std::string>("electronQuality");
  version_ = conf.getParameter<std::string>("electronVersion");
  
  if (type_ == "robust" || type_ == "classbased") {
    if (quality_ == "loose" || quality_ == "tight" || quality_ == "highenergy" ) {
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
  double pin  = electron->trackMomentumAtVtx().R(); 
  double pout = electron->trackMomentumOut().R(); 
  double fBrem = (pin-pout)/pin;
  
  int cat;
  if((electron->isEB() && fBrem<0.06) || (electron->isEE() && fBrem<0.1)) 
    cat=1;
  else if (eOverP < 1.2 && eOverP > 0.8) 
    cat=0;
  else 
    cat=2;
  
  return cat;
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
  std::vector<float> vCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  //std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;
  double sigmaee = sqrt(vCov[0]);
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();
  
  int eb;
  if (electron->isEB()) 
    eb = 0;
  else {
    eb = 1; 
    sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  std::vector<double> cut;
    
  // ROBUST Selection
  if (type_ == "robust") {

    // hoe, sigmaEtaEta, dPhiIn, dEtaIn
    if (electron->isEB())
      cut = cuts_.getParameter<std::vector<double> >("barrel");
    else
      cut = cuts_.getParameter<std::vector<double> >("endcap");

    if (hOverE > cut[0]) 
      return 0.;    

    if (sigmaee > cut[1]) 
      return 0.;    

    if (fabs(deltaPhiIn) > cut[2]) 
      return 0.;    

    if (fabs(deltaEtaIn) > cut[3]) 
      return 0.;    
    
    return 1.;
  }
  
  int cat = classify(electron);

  // LOOSE and TIGHT Selections
  if (type_ == "classbased") {
    
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
  
  return 0.;
}
