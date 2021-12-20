#ifndef GeneratorInterface_Pythia8Interface_SuepDecay_h
#define GeneratorInterface_Pythia8Interface_SuepDecay_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/SuepShower.h"
#include <memory>
#include "GeneratorInterface/Pythia8Interface/interface/CustomHook.h"

// Adapted by Kevin Pedro to run on cmssw as a user hook
class SuepDecay : public Pythia8::UserHooks {
public:
  SuepDecay(const edm::ParameterSet& iConfig);
  ~SuepDecay() override {}

  bool initAfterBeams() override;

  bool canVetoProcessLevel() override { return true; }
  bool doVetoProcessLevel(Pythia8::Event& event) override;

protected:
  int idMediator_, idDark_;
  float temperature_, mDark_;
  std::unique_ptr<SuepShower> suep_shower_;
};

REGISTER_USERHOOK(SuepDecay);
#endif
