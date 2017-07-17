//
// L1FastjetCorrector
// ------------------
//
// 08/09/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//
#ifndef JetMETCorrections_JetCorrector_L1FastjetCorrectorImpl_h
#define JetMETCorrections_JetCorrector_L1FastjetCorrectorImpl_h 1

#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class ConfigurationDescriptions;
}

class L1FastjetCorrectorImplMaker : public JetCorrectorImplMakerBase {
 public:
  L1FastjetCorrectorImplMaker(edm::ParameterSet const&, edm::ConsumesCollector);
  std::unique_ptr<reco::JetCorrectorImpl> make(edm::Event const&, edm::EventSetup const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions);
 private:
  edm::EDGetTokenT<double> rhoToken_;
};

class L1FastjetCorrectorImpl : public reco::JetCorrectorImpl
{
public:
  typedef L1FastjetCorrectorImplMaker Maker;

  // construction / destruction
  L1FastjetCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector, double rho):
  rho_(rho),corrector_(corrector) {}
  
  //member functions
  
  /// apply correction using Jet information only
  virtual double correction(const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction(const reco::Jet& fJet) const;

  //----- if correction needs a jet reference -------------
  virtual bool refRequired() const { return false; }

private:
  // member data
  double rho_;
  std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector_;
};

#endif
