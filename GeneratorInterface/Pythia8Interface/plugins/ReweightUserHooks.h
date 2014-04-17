#include "Pythia8/Pythia.h"
#include "TF1.h"

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


class RapReweightUserHook : public Pythia8::UserHooks
{
  public:
    RapReweightUserHook(const std::string& _yLabsigma_func, double _yLab_power,
                        const std::string& _yCMsigma_func, double _yCM_power,
                        double _pTHatMin, double _pTHatMax) :
      yLabsigma_func(_yLabsigma_func), yCMsigma_func(_yCMsigma_func),
      yLab_power(_yLab_power), yCM_power(_yCM_power),
      pTHatMin(_pTHatMin), pTHatMax(_pTHatMax)
    {
      // empirical parametrizations defined in configuration file
      yLabsigma = TF1("yLabsigma", yLabsigma_func.c_str(), pTHatMin, pTHatMax);
      yCMsigma = TF1("yCMsigma", yLabsigma_func.c_str(), pTHatMin, pTHatMax);
    }
    virtual ~RapReweightUserHook() {}

    virtual bool canBiasSelection() { return true; }

    virtual double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                      const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
    {
      //the variable selBias of the base class should be used;
      if ((sigmaProcessPtr->nFinal() == 2)) {
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double yLab = 0.5*log(x1/x2);
        double yCM = 0.5*log( phaseSpacePtr->tHat() / phaseSpacePtr->uHat() );
        double pTHat = phaseSpacePtr->pTHat();
        double sigmaLab = yLabsigma.Eval(pTHat);
        double sigmaCM = yCMsigma.Eval(pTHat);
        // empirical reweighting function
        selBias = exp( pow(fabs(yLab),yLab_power)/(2*sigmaLab*sigmaLab) +
                       pow(fabs(yCM),yCM_power)/(2*sigmaCM*sigmaCM) );
        return selBias;
      }
      selBias = 1.;
      return selBias;
    }

  private:
    std::string yLabsigma_func, yCMsigma_func;
    double yLab_power, yCM_power, pTHatMin, pTHatMax;
    TF1 yLabsigma, yCMsigma;
};


class PtHatRapReweightUserHook : public Pythia8::UserHooks
{
  public:
    PtHatRapReweightUserHook(const std::string& _yLabsigma_func, double _yLab_power,
                             const std::string& _yCMsigma_func, double _yCM_power,
                             double _pTHatMin, double _pTHatMax,
                             double _pt = 15, double _power = 4.5) :
      yLabsigma_func(_yLabsigma_func), yCMsigma_func(_yCMsigma_func),
      yLab_power(_yLab_power), yCM_power(_yCM_power),
      pTHatMin(_pTHatMin), pTHatMax(_pTHatMax), pt(_pt), power(_power)
    {
      // empirical parametrizations defined in configuration file
      yLabsigma = TF1("yLabsigma", yLabsigma_func.c_str(), pTHatMin, pTHatMax);
      yCMsigma = TF1("yCMsigma", yLabsigma_func.c_str(), pTHatMin, pTHatMax);
    }
    virtual ~PtHatRapReweightUserHook() {}

    virtual bool canBiasSelection() { return true; }

    virtual double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                      const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
    {
      //the variable selBias of the base class should be used;
      if ((sigmaProcessPtr->nFinal() == 2)) {
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double yLab = 0.5*log(x1/x2);
        double yCM = 0.5*log( phaseSpacePtr->tHat() / phaseSpacePtr->uHat() );
        double pTHat = phaseSpacePtr->pTHat();
        double sigmaLab = yLabsigma.Eval(pTHat);
        double sigmaCM = yCMsigma.Eval(pTHat);
        // empirical reweighting function
        selBias = pow(pTHat / pt, power) * exp( pow(fabs(yLab),yLab_power)/(2*sigmaLab*sigmaLab) + 
                  pow(fabs(yCM),yCM_power)/(2*sigmaCM*sigmaCM) );
        return selBias;
      }
      selBias = 1.;
      return selBias;
    }

  private:
    std::string yLabsigma_func, yCMsigma_func;
    double yLab_power, yCM_power, pTHatMin, pTHatMax, pt, power;
    TF1 yLabsigma, yCMsigma;
};
