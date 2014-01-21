#include "Pythia8/Pythia.h"

class PtHatReweightUserHook : public Pythia8::UserHooks
{
  public:
    PtHatReweightUserHook(double _pt = 15, double _power = 4.5) :
      pt(_pt), power(_power) {}
    virtual ~PtHatReweightUserHook() {}

    virtual bool canBiasSelection() { return true; }

    virtual double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                      const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
    {
      //the variable selBias of the base class should be used;
      if ((sigmaProcessPtr->nFinal() == 2)) {
        selBias = pow(phaseSpacePtr->pTHat() / pt, power);
        return selBias;
      }
      selBias = 1.;
      return selBias;
    }

  private:
    double pt, power;
};
