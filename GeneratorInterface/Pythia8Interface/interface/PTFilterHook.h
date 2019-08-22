
#include "Pythia8/UserHooks.h"

class PTFilterHook : public Pythia8::UserHooks {
public:
  // Constructor and destructor.
  PTFilterHook() {}

  //--------------------------------------------------------------------------

  bool initAfterBeams() override;
  bool canVetoPT() override { return true; }
  double scaleVetoPT() override { return scale_; }
  bool doVetoPT(int iPos, const Pythia8::Event& event) override { return checkVetoPT(iPos, event); }
  bool checkVetoPT(int iPos, const Pythia8::Event& event);

  //--------------------------------------------------------------------------

private:
  bool filter_;
  int quark_;
  double scale_;
  double quarkY_;
  double quarkPt_;
};
