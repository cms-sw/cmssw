// L1Offset jet corrector class. Inherits from JetCorrector.h
#ifndef L1JPTOffsetCorrectorImpl_h
#define L1JPTOffsetCorrectorImpl_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class ConfigurationDescriptions;
}

namespace reco {
  class JetCorrector;
}

class L1JPTOffsetCorrectorImplMaker : public JetCorrectorImplMakerBase {
 public:
  L1JPTOffsetCorrectorImplMaker(edm::ParameterSet const&, edm::ConsumesCollector);
  std::unique_ptr<reco::JetCorrectorImpl> make(edm::Event const&, edm::EventSetup const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions);
 private:
  edm::EDGetTokenT<reco::JetCorrector> offsetCorrectorToken_;
  bool useOffset_;
};

//----- LXXXCorrector interface -------------------------------
class L1JPTOffsetCorrectorImpl : public reco::JetCorrectorImpl 
{
  public:
  typedef L1JPTOffsetCorrectorImplMaker Maker;

    //----- constructors---------------------------------------
  L1JPTOffsetCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector,
			   const reco::JetCorrector* offsetService);   

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const override;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const override;

    //----- if correction needs event information -------------
    virtual bool refRequired() const override {return false;}

  private:
    //----- member data ---------------------------------------
    const reco::JetCorrector* offsetService_;
    std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector_;
};

#endif
