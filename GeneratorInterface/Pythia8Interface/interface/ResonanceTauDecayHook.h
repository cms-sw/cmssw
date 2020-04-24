#include "Pythia8/UserHooks.h"
#include "Pythia8/Event.h"
#include "Pythia8/TauDecays.h"

class ResonanceTauDecayHook : public Pythia8::UserHooks {

public:

  ResonanceTauDecayHook() {
    decayer = Pythia8::TauDecays();
  }

  bool initAfterBeams() override;
  // Allow a veto for the process level, to gain access to decays.
  bool canVetoProcessLevel() override {return true;}
  // Access the event after resonance decays.
  bool doVetoProcessLevel(Pythia8::Event& process) override {return checkResonanceTauDecays(process); }
  bool checkResonanceTauDecays(Pythia8::Event& process);

private:

  Pythia8::TauDecays decayer;
  bool filter_;

};
