#include "RecoEgamma/ElectronIdentification/interface/PTDRElectronID.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

void PTDRElectronID::setup(const edm::ParameterSet& conf) {

  // Get all the parameters
  baseSetup(conf);
  
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
    edm::LogError("PTDRElectronID") << "Invalid electronQuality parameter: must be tight, medium or loose." ;
    exit (1);
  }
  
}

double PTDRElectronID::result(const reco::GsfElectron* electron,
                              const edm::Event& e) {

  //determine which element of the cut arrays in electronId.cfi to read
  //depending on the electron classification
  int icut=0;
  switch (electron->classification()) {
  case 0:   icut=0; break;
  case 10:  icut=1; break;
  case 20:  icut=2; break;
  case 30:  icut=3; break;
  case 31:  icut=3; break;
  case 32:  icut=3; break;
  case 33:  icut=3; break;
  case 34:  icut=3; break;
  case 40:  icut=8; break;
  case 100: icut=4; break;
  case 110: icut=5; break;
  case 120: icut=6; break;
  case 130: icut=7; break;
  case 131: icut=7; break;
  case 132: icut=7; break;
  case 133: icut=7; break;
  case 134: icut=7; break;

  default:
    edm::LogError("CutBasedElectronID") << "Error: unrecognized electron classification ";
    break;
  }

  if (acceptCracks_[variables_])
    if (electron->classification()==40) return 1.;
  
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
    if (fabs(value)>maxcut[icut]) return 0.;
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

  const reco::ClusterShape& cluShape = getClusterShape(electron,e);

  if (useE9overE25_[variables_]) {
    double value = (cluShape).e3x3()/(cluShape).e5x5();
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("E9overE25");
    if (fabs(value)<mincut[icut]) return 0.;
  }

  if (useSigmaEtaEta_[variables_]) {
    double value = (cluShape).covEtaEta();
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("sigmaEtaEtaMax");
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("sigmaEtaEtaMin");
    if (sqrt(value)<mincut[icut] || sqrt(value)>maxcut[icut]) return 0.;
  }

  if (useSigmaPhiPhi_[variables_]) {
    double value = (cluShape).covPhiPhi();
    std::vector<double> mincut = cuts_.getParameter<std::vector<double> >("sigmaPhiPhiMin");
    std::vector<double> maxcut = cuts_.getParameter<std::vector<double> >("sigmaPhiPhiMax");
    if (sqrt(value)<mincut[icut] || sqrt(value)>maxcut[icut]) return 0.;
  }

  return 1.;

}
