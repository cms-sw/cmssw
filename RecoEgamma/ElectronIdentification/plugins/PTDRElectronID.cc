#include "PTDRElectronID.h"

PTDRElectronID::PTDRElectronID(const edm::ParameterSet& conf) : cuts_{conf} {}

PTDRElectronID::Cuts::Cuts(const edm::ParameterSet& conf) {
  int variables = 0;
  auto quality = conf.getParameter<std::string>("electronQuality");
  edm::ParameterSet cuts;
  if (quality == "tight") {
    cuts = conf.getParameter<edm::ParameterSet>("tightEleIDCuts");
    variables = 2;
  } else if (quality == "medium") {
    cuts = conf.getParameter<edm::ParameterSet>("mediumEleIDCuts");
    variables = 1;
  } else if (quality == "loose") {
    cuts = conf.getParameter<edm::ParameterSet>("looseEleIDCuts");
    variables = 0;
  } else {
    throw cms::Exception("Configuration")
        << "Invalid electronQuality parameter in PTDElectronID: must be tight, medium or loose.";
  }

  useEoverPIn_ = conf.getParameter<std::vector<int> >("useEoverPIn").at(variables);
  if (useEoverPIn_) {
    EoverPInMax_ = cuts.getParameter<std::vector<double> >("EoverPInMax");
    EoverPInMin_ = cuts.getParameter<std::vector<double> >("EoverPInMin");
  }
  useDeltaEtaIn_ = conf.getParameter<std::vector<int> >("useDeltaEtaIn").at(variables);
  if (useDeltaEtaIn_) {
    deltaEtaIn_ = cuts.getParameter<std::vector<double> >("deltaEtaIn");
  }
  useDeltaPhiIn_ = conf.getParameter<std::vector<int> >("useDeltaPhiIn").at(variables);
  if (useDeltaPhiIn_) {
    deltaPhiIn_ = cuts.getParameter<std::vector<double> >("deltaPhiIn");
  }
  useHoverE_ = conf.getParameter<std::vector<int> >("useHoverE").at(variables);
  if (useHoverE_) {
    HoverE_ = cuts.getParameter<std::vector<double> >("HoverE");
  }
  useE9overE25_ = conf.getParameter<std::vector<int> >("useE9overE25").at(variables);
  if (useE9overE25_) {
    E9overE25_ = cuts.getParameter<std::vector<double> >("E9overE25");
  }
  useEoverPOut_ = conf.getParameter<std::vector<int> >("useEoverPOut").at(variables);
  if (useEoverPOut_) {
    EoverPOutMax_ = cuts.getParameter<std::vector<double> >("EoverPOutMax");
    EoverPOutMin_ = cuts.getParameter<std::vector<double> >("EoverPOutMin");
  }
  useDeltaPhiOut_ = conf.getParameter<std::vector<int> >("useDeltaPhiOut").at(variables);
  if (useDeltaPhiOut_) {
    deltaPhiOut_ = cuts.getParameter<std::vector<double> >("deltaPhiOut");
  }
  useInvEMinusInvP_ = conf.getParameter<std::vector<int> >("useInvEMinusInvP").at(variables);
  if (useInvEMinusInvP_) {
    invEMinusInvP_ = cuts.getParameter<std::vector<double> >("invEMinusInvP");
  }
  useBremFraction_ = conf.getParameter<std::vector<int> >("useBremFraction").at(variables);
  if (useBremFraction_) {
    bremFraction_ = cuts.getParameter<std::vector<double> >("bremFraction");
  }
  useSigmaEtaEta_ = conf.getParameter<std::vector<int> >("useSigmaEtaEta").at(variables);
  if (useSigmaEtaEta_) {
    sigmaEtaEtaMax_ = cuts.getParameter<std::vector<double> >("sigmaEtaEtaMax");
    sigmaEtaEtaMin_ = cuts.getParameter<std::vector<double> >("sigmaEtaEtaMin");
  }
  useSigmaPhiPhi_ = conf.getParameter<std::vector<int> >("useSigmaPhiPhi").at(variables);
  if (useSigmaPhiPhi_) {
    sigmaPhiPhiMin_ = cuts.getParameter<std::vector<double> >("sigmaPhiPhiMin");
    sigmaPhiPhiMax_ = cuts.getParameter<std::vector<double> >("sigmaPhiPhiMax");
  }
  acceptCracks_ = conf.getParameter<std::vector<int> >("acceptCracks").at(variables);
}

double PTDRElectronID::result(const reco::GsfElectron* electron, const edm::Event& e, const edm::EventSetup& es) const {
  //determine which element of the cut arrays in cfi file to read
  //depending on the electron classification
  int icut = 0;
  int elClass = electron->classification();
  if (electron->isEB())  //barrel
  {
    if (elClass == reco::GsfElectron::GOLDEN)
      icut = 0;
    if (elClass == reco::GsfElectron::BIGBREM)
      icut = 1;
    //if (elClass == reco::GsfElectron::NARROW)    icut=2;
    if (elClass == reco::GsfElectron::SHOWERING)
      icut = 3;
    if (elClass == reco::GsfElectron::GAP)
      icut = 8;
  }
  if (electron->isEE())  //endcap
  {
    if (elClass == reco::GsfElectron::GOLDEN)
      icut = 4;
    if (elClass == reco::GsfElectron::BIGBREM)
      icut = 5;
    //if (elClass == reco::GsfElectron::NARROW)    icut=6;
    if (elClass == reco::GsfElectron::SHOWERING)
      icut = 7;
    if (elClass == reco::GsfElectron::GAP)
      icut = 8;
  }
  if (elClass == reco::GsfElectron::UNKNOWN) {
    edm::LogError("PTDRElectronID") << "Error: unrecognized electron classification ";
    return 1.;
  }

  if (cuts_.acceptCracks_)
    if (elClass == reco::GsfElectron::GAP)
      return 1.;

  if (cuts_.useEoverPIn_) {
    double value = electron->eSuperClusterOverP();
    std::vector<double> const& maxcut = cuts_.EoverPInMax_;
    std::vector<double> const& mincut = cuts_.EoverPInMin_;
    if (value < mincut[icut] || value > maxcut[icut])
      return 0.;
  }

  if (cuts_.useDeltaEtaIn_) {
    double value = electron->deltaEtaSuperClusterTrackAtVtx();
    std::vector<double> const& maxcut = cuts_.deltaEtaIn_;
    if (fabs(value) > maxcut[icut])
      return 0.;
  }

  if (cuts_.useDeltaPhiIn_) {
    double value = electron->deltaPhiSuperClusterTrackAtVtx();
    std::vector<double> const& maxcut = cuts_.deltaPhiIn_;
    if (fabs(value) > maxcut[icut])
      return 0.;
  }

  if (cuts_.useHoverE_) {
    double value = electron->hadronicOverEm();
    std::vector<double> const& maxcut = cuts_.HoverE_;
    if (value > maxcut[icut])
      return 0.;
  }

  if (cuts_.useEoverPOut_) {
    double value = electron->eSeedClusterOverPout();
    std::vector<double> const& maxcut = cuts_.EoverPOutMax_;
    std::vector<double> const& mincut = cuts_.EoverPOutMin_;
    if (value < mincut[icut] || value > maxcut[icut])
      return 0.;
  }

  if (cuts_.useDeltaPhiOut_) {
    double value = electron->deltaPhiSeedClusterTrackAtCalo();
    std::vector<double> const& maxcut = cuts_.deltaPhiOut_;
    if (fabs(value) > maxcut[icut])
      return 0.;
  }

  if (cuts_.useInvEMinusInvP_) {
    double value = (1. / electron->caloEnergy()) - (1. / electron->trackMomentumAtVtx().R());
    std::vector<double> const& maxcut = cuts_.invEMinusInvP_;
    if (value > maxcut[icut])
      return 0.;
  }

  if (cuts_.useBremFraction_) {
    double value = electron->trackMomentumAtVtx().R() - electron->trackMomentumOut().R();
    std::vector<double> const& mincut = cuts_.bremFraction_;
    if (value < mincut[icut])
      return 0.;
  }

  //EcalClusterLazyTools lazyTools = getClusterShape(e,es);
  //std::vector<float> vCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  //std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;

  if (cuts_.useE9overE25_) {
    double value = electron->r9() * electron->superCluster()->energy() / electron->e5x5();
    std::vector<double> const& mincut = cuts_.E9overE25_;
    if (fabs(value) < mincut[icut])
      return 0.;
  }

  if (cuts_.useSigmaEtaEta_) {
    std::vector<double> const& maxcut = cuts_.sigmaEtaEtaMax_;
    std::vector<double> const& mincut = cuts_.sigmaEtaEtaMin_;
    if (electron->sigmaIetaIeta() < mincut[icut] || electron->sigmaIetaIeta() > maxcut[icut])
      return 0.;
  }

  if (cuts_.useSigmaPhiPhi_) {
    std::vector<double> const& mincut = cuts_.sigmaPhiPhiMin_;
    std::vector<double> const& maxcut = cuts_.sigmaPhiPhiMax_;
    if (electron->sigmaIphiIphi() < mincut[icut] || electron->sigmaIphiIphi() > maxcut[icut])
      return 0.;
  }

  return 1.;
}
