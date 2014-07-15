#include "RecoEgamma/ElectronIdentification/interface/PTDRElectronID.h"

void PTDRElectronID::setup(const edm::ParameterSet& conf) {

  // Get all the parameters
  //baseSetup(conf);
  
  quality_ =  conf.getParameter<std::string>("electronQuality");
  
  useEoverPIn_ = conf.getParameter<std::vector<int> >("useEoverPIn");
  useDeltaEtaIn_ = conf.getParameter<std::vector<int> >("useDeltaEtaIn");
  useDeltaPhiIn_ = conf.getParameter<std::vector<int> >("useDeltaPhiIn");
  useHoverE_ = conf.getParameter<std::vector<int> >("useHoverE");
  useE9overE25_ = conf.getParameter<std::vector<int> >("useE9overE25");
  useEoverPOut_ = conf.getParameter<std::vector<int> >("useEoverPOut");
  useDeltaPhiOut_ = conf.getParameter<std::vector<int> >("useDeltaPhiOut");
  useInvEMinusInvP_ = conf.getParameter<std::vector<int> >("useInvEMinusInvP");
  useBremFraction_ = conf.getParameter<std::vector<int> >("useBremFraction");
  useSigmaEtaEta_ = conf.getParameter<std::vector<int> >("useSigmaEtaEta");
  useSigmaPhiPhi_ = conf.getParameter<std::vector<int> >("useSigmaPhiPhi");
  acceptCracks_ = conf.getParameter<std::vector<int> >("acceptCracks");
  
  if (quality_=="tight") {
    cuts_ = conf.getParameter<edm::ParameterSet>("tightEleIDCuts");
    variables_ = 2 ;
  } else if (quality_=="medium") {
    cuts_ = conf.getParameter<edm::ParameterSet>("mediumEleIDCuts");
    variables_ = 1 ;
  } else if (quality_=="loose") {
    cuts_ = conf.getParameter<edm::ParameterSet>("looseEleIDCuts");
    variables_ = 0 ;
  } else {
     throw cms::Exception("Configuration") << "Invalid electronQuality parameter in PTDElectronID: must be tight, medium or loose." ;
  }  
}

double PTDRElectronID::result(const reco::GsfElectron* electron,
                              const edm::Event& e ,
                              const edm::EventSetup& es) {

  //determine which element of the cut arrays in cfi file to read
  //depending on the electron classification
  int icut=0;
  int elClass = electron->classification() ;
  if (electron->isEB()) //barrel
     {
       if (elClass == reco::GsfElectron::GOLDEN)    icut=0;
       if (elClass == reco::GsfElectron::BIGBREM)   icut=1;
       //if (elClass == reco::GsfElectron::NARROW)    icut=2;
       if (elClass == reco::GsfElectron::SHOWERING) icut=3;
       if (elClass == reco::GsfElectron::GAP)       icut=8;
     }
  if (electron->isEE()) //endcap
     {
       if (elClass == reco::GsfElectron::GOLDEN)    icut=4;
       if (elClass == reco::GsfElectron::BIGBREM)   icut=5;
       //if (elClass == reco::GsfElectron::NARROW)    icut=6;
       if (elClass == reco::GsfElectron::SHOWERING) icut=7;
       if (elClass == reco::GsfElectron::GAP)       icut=8;
     }
  if (elClass == reco::GsfElectron::UNKNOWN) 
     {
       edm::LogError("PTDRElectronID") << "Error: unrecognized electron classification ";
       return 1.;
     }

  if (acceptCracks_[variables_])
    if (elClass == reco::GsfElectron::GAP) return 1.;
  
  if (useEoverPIn_[variables_]) {
    double value = electron->eSuperClusterOverP();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("EoverPInMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("EoverPInMin");
    if (value<mincut[icut] || value>maxcut[icut]) return 0.;
  }

  if (useDeltaEtaIn_[variables_]) {
    double value = electron->deltaEtaSuperClusterTrackAtVtx();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("deltaEtaIn");
    if (fabs(value)>maxcut[icut]) return 0.;
  }

  if (useDeltaPhiIn_[variables_]) {
    double value = electron->deltaPhiSuperClusterTrackAtVtx();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("deltaPhiIn");
    if (fabs(value)>maxcut[icut]) return 0.;
  }

  if (useHoverE_[variables_]) {
    double value = electron->hadronicOverEm();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("HoverE");
    if (value>maxcut[icut]) return 0.;
  }

  if (useEoverPOut_[variables_]) {
    double value = electron->eSeedClusterOverPout();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("EoverPOutMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("EoverPOutMin");
    if (value<mincut[icut] || value>maxcut[icut]) return 0.;
  }

  if (useDeltaPhiOut_[variables_]) {
    double value = electron->deltaPhiSeedClusterTrackAtCalo();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("deltaPhiOut");
    if (fabs(value)>maxcut[icut]) return 0.;
  }

  if (useInvEMinusInvP_[variables_]) {
    double value = (1./electron->caloEnergy())-(1./electron->trackMomentumAtVtx().R());
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("invEMinusInvP");
    if (value>maxcut[icut]) return 0.;
  }

  if (useBremFraction_[variables_]) {
    double value = electron->trackMomentumAtVtx().R()-electron->trackMomentumOut().R();
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("bremFraction");
    if (value<mincut[icut]) return 0.;
  }

  //EcalClusterLazyTools lazyTools = getClusterShape(e,es);
  //std::vector<float> vCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  //std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;
    
  if (useE9overE25_[variables_]) {
    double value = electron->r9()*electron->superCluster()->energy()/electron->e5x5();
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("E9overE25");
    if (fabs(value)<mincut[icut]) return 0.;
  }

  if (useSigmaEtaEta_[variables_]) {
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("sigmaEtaEtaMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("sigmaEtaEtaMin");
    if (electron->sigmaIetaIeta()<mincut[icut] || electron->sigmaIetaIeta()>maxcut[icut]) return 0.;
  }

  if (useSigmaPhiPhi_[variables_]) {
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("sigmaPhiPhiMin");
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("sigmaPhiPhiMax");
    if (electron->sigmaIphiIphi()<mincut[icut] || electron->sigmaIphiIphi()>maxcut[icut]) return 0.;
  }

  return 1.;

}
