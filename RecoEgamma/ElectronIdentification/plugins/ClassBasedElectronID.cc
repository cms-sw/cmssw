#include "ClassBasedElectronID.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  edm::ParameterSet fromQuality(const edm::ParameterSet& conf) {
    auto quality = conf.getParameter<std::string>("electronQuality");

    if (quality == "Eff95Cuts") {
      return conf.getParameter<edm::ParameterSet>("Eff95Cuts");
    }

    else if (quality == "Eff90Cuts") {
      return conf.getParameter<edm::ParameterSet>("Eff90Cuts");
    }

    throw cms::Exception("ClassBasedElectronID")
        << "Invalid electronQuality parameter: must be tight, medium or loose.";
  }
}  // namespace
// ===========================================================================================================
ClassBasedElectronID::ClassBasedElectronID(const edm::ParameterSet& conf)
    : cuts_{fromQuality(conf)}
// ===========================================================================================================
{}  // end of setup

ClassBasedElectronID::Cuts::Cuts(const edm::ParameterSet& conf) {
  deltaEtaIn_ = conf.getParameter<std::vector<double> >("deltaEtaIn");
  sigmaIetaIetaMax_ = conf.getParameter<std::vector<double> >("sigmaIetaIetaMax");
  sigmaIetaIetaMin_ = conf.getParameter<std::vector<double> >("sigmaIetaIetaMin");
  HoverE_ = conf.getParameter<std::vector<double> >("HoverE");
  EoverPOutMax_ = conf.getParameter<std::vector<double> >("EoverPOutMax");
  EoverPOutMin_ = conf.getParameter<std::vector<double> >("EoverPOutMin");
  deltaPhiInChargeMax_ = conf.getParameter<std::vector<double> >("deltaPhiInChargeMax");
  deltaPhiInChargeMin_ = conf.getParameter<std::vector<double> >("deltaPhiInChargeMin");
}

double ClassBasedElectronID::result(const reco::GsfElectron* electron,
                                    const edm::Event& e,
                                    const edm::EventSetup& es) const {
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
    if (elClass == reco::GsfElectron::SHOWERING)
      icut = 2;
    if (elClass == reco::GsfElectron::GAP)
      icut = 6;
  }
  if (electron->isEE())  //endcap
  {
    if (elClass == reco::GsfElectron::GOLDEN)
      icut = 3;
    if (elClass == reco::GsfElectron::BIGBREM)
      icut = 4;
    if (elClass == reco::GsfElectron::SHOWERING)
      icut = 5;
    if (elClass == reco::GsfElectron::GAP)
      icut = 7;
  }
  if (elClass == reco::GsfElectron::UNKNOWN) {
    edm::LogError("ClassBasedElectronID") << "Error: unrecognized electron classification ";
    return 1.;
  }

  constexpr bool useDeltaEtaIn = true;
  constexpr bool useSigmaIetaIeta = true;
  constexpr bool useHoverE = true;
  constexpr bool useEoverPOut = true;
  constexpr bool useDeltaPhiInCharge = true;

  // DeltaEtaIn
  if (useDeltaEtaIn) {
    double value = electron->deltaEtaSuperClusterTrackAtVtx();
    std::vector<double> const& maxcut = cuts_.deltaEtaIn_;
    if (fabs(value) > maxcut[icut])
      return 0.;
  }

  // SigmaIetaIeta
  if (useSigmaIetaIeta) {
    double value = electron->sigmaIetaIeta();
    std::vector<double> const& maxcut = cuts_.sigmaIetaIetaMax_;
    std::vector<double> const& mincut = cuts_.sigmaIetaIetaMin_;
    if (value < mincut[icut] || value > maxcut[icut])
      return 0.;
  }

  // H/E
  if (useHoverE) {  //_[variables_]) {
    double value = electron->hadronicOverEm();
    std::vector<double> const& maxcut = cuts_.HoverE_;
    if (value > maxcut[icut])
      return 0.;
  }  // if use

  // Eseed/Pout
  if (useEoverPOut) {
    double value = electron->eSeedClusterOverPout();
    std::vector<double> maxcut = cuts_.EoverPOutMax_;
    std::vector<double> mincut = cuts_.EoverPOutMin_;
    if (value < mincut[icut] || value > maxcut[icut])
      return 0.;
  }

  // DeltaPhiIn*Charge
  if (useDeltaPhiInCharge) {
    double value1 = electron->deltaPhiSuperClusterTrackAtVtx();
    double value2 = electron->charge();
    double value = value1 * value2;
    std::vector<double> maxcut = cuts_.deltaPhiInChargeMax_;
    std::vector<double> mincut = cuts_.deltaPhiInChargeMin_;
    if (value < mincut[icut] || value > maxcut[icut])
      return 0.;
  }

  return 1.;
}
