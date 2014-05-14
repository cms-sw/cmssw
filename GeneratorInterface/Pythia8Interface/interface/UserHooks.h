#include <Pythia.h>

class PtHatReweightUserHook : public Pythia8::UserHooks
{
public:
  PtHatReweightUserHook(double _pt = 15, double _power = 4.5) :
                         pt(_pt), power(_power), factor(1.) {}
  virtual ~PtHatReweightUserHook() {}

  virtual bool canModifySigma() { return true; }

  virtual double multiplySigmaBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                       const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
  {
    if ((sigmaProcessPtr->nFinal() == 2) && inEvent) {
      factor = pow(phaseSpacePtr->pTHat() / pt, power);
      return factor;
    }
    factor = 1;
    return factor;
  }

  double getFactor() {return factor;}

private:
	double pt, power;
        double factor;
};
