
#include "Pythia8/UserHooks.h"
#include "Pythia8/Event.h"

class ResonanceDecayFilterHook : public Pythia8::UserHooks {

public:  

  // Constructor and destructor.
  ResonanceDecayFilterHook() {}
                    
//--------------------------------------------------------------------------

  bool initAfterBeams() override;
  bool canVetoResonanceDecays() override { return true; }
  bool doVetoResonanceDecays(Pythia8::Event& process) override { return checkVetoResonanceDecays(process); }
  bool checkVetoResonanceDecays(const Pythia8::Event& process);

//--------------------------------------------------------------------------

private:
  bool filter_;
  bool exclusive_;
  bool eMuAsEquivalent_;
  bool eMuTauAsEquivalent_;
  bool allNuAsEquivalent_;
  bool udscAsEquivalent_;
  bool udscbAsEquivalent_;
  std::vector<int> mothers_;
  std::vector<int> daughters_;
  
  std::map<int,int> requestedDaughters_;
  std::map<int,int> observedDaughters_;
  
};
