#include <Pythia.h>

class PtHatReweightUserHook : public Pythia8::UserHooks
{
public:
  PtHatReweightUserHook(double _pt = 15, double _power = 4.5) :
                         pt(_pt), power(_power) {}
  virtual ~PtHatReweightUserHook() {}

  virtual bool canModifySigma() { return true; }

  virtual double multiplySigmaBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                       const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
  {
    if ((sigmaProcessPtr->nFinal() == 2) && inEvent)
      return pow(phaseSpacePtr->pTHat() / pt, power);
    return 1;
  }
private:
	double pt, power;
};
