// Generic LX jet corrector class. Inherits from JetCorrector.h
#ifndef LXXXCorrectorImpl_h
#define LXXXCorrectorImpl_h

#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class ConfigurationDescriptions;
}
class FactorizedJetCorrector;

class LXXXCorrectorImplMaker : public JetCorrectorImplMakerBase {
 public:
  LXXXCorrectorImplMaker(edm::ParameterSet const&, edm::ConsumesCollector);
  std::unique_ptr<reco::JetCorrectorImpl> make(edm::Event const&, edm::EventSetup const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions);
};

//----- LXXXCorrectorImpl interface -------------------------------
class LXXXCorrectorImpl : public reco::JetCorrectorImpl
{
  public:
    typedef LXXXCorrectorImplMaker Maker;

    //----- constructors---------------------------------------
    LXXXCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> calculator, unsigned int level);

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const override;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const override;

    //----- if correction needs a jet reference -------------
    virtual bool refRequired() const override { return false; }

  private:
    //----- member data ---------------------------------------
    unsigned int mLevel;
    std::shared_ptr<FactorizedJetCorrectorCalculator const> mCorrector;
};

#endif
