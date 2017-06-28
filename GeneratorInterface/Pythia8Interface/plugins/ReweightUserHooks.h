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

class PtHatEmpReweightUserHook : public Pythia8::UserHooks
{
  public:
   PtHatEmpReweightUserHook(double _pTHatMin, double _pTHatMax) :
      pTHatMin(_pTHatMin), pTHatMax(_pTHatMax) {
      // Normalized to Event/fb-1
      //Mikko
      //sigma = TF1("sigma", "max(2.e-15,[0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1]))", pTHatMin, pTHatMax);
      //Line
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(1.99851e-11+-3.07464e-15*x))", pTHatMin, pTHatMax);
      //quartic function
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(-3.07464e-18*pow(x-5250,2)+(1.436e-12)+(2e-27*pow(x,4))))", pTHatMin, pTHatMax);
      //sigmoid
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(0.00001+(TMath::Exp(0.015*(x-5500))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(1.0+(TMath::Exp(0.035*(x-5800.0))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(1.0+(TMath::Exp(0.017*(x-5600.0))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(0.5507+(TMath::Exp(0.016*(x-5550.0))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(0.73713+(TMath::Exp(0.0167*(x-5580.0))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(0.03624+(TMath::Exp(0.015*(x-5530.0))))))", pTHatMin, pTHatMax);
      //sigma = TF1("sigma", "((x<=5500)*([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1])))+((x>5500)*(3.07464e-12/(1.0+(TMath::Exp(0.022*(x-5680.0))))))", pTHatMin, pTHatMax);
      std::string fLT5500 = "([0]*pow(x,[2]+[3]*log(0.01*x)+[4]*pow(log(0.01*x),2))*pow(1-2*x/13000.,[1]))";
      std::string f5500to5700 = "(3.07464e-12/(0.5507+(TMath::Exp(0.016*(x-5550.0)))))";
      std::string f5700to5800 = "(3.07464e-12/(0.73713+(TMath::Exp(0.0167*(x-5580.0)))))";
      std::string f5800to6000 = "(3.07464e-12/(1.0+(TMath::Exp(0.017*(x-5600.0)))))";
      std::string f6000toInf = "(3.07464e-12/(1.0+(TMath::Exp(0.023*(x-5705.0)))))";
      const char *fmt = "((x<=5500)*%s)+((x>5500)*(x<=5700)*%s)+((x>5700)*(x<=5800)*%s)+((x>5800)*(x<=6000)*%s)+((x>6000)*%s)";
      int sz = std::snprintf(nullptr, 0, fmt, fLT5500.c_str(), f5500to5700.c_str(), f5700to5800.c_str(), f5800to6000.c_str(), f6000toInf.c_str());
      std::vector<char> buf(sz + 1); // +1 for null terminator
      std::snprintf(&buf[0], buf.size(), fmt, fLT5500.c_str(), f5500to5700.c_str(), f5700to5800.c_str(), f5800to6000.c_str(), f6000toInf.c_str());
      sigma = TF1("sigma", &buf[0], pTHatMin, pTHatMax);
      sigma.SetParameters(5.66875e+13,9.93446e+00,-3.96986e+00,-2.26690e-01,1.75926e-02);
    }
    virtual ~PtHatEmpReweightUserHook() {}

    virtual bool canBiasSelection() { return true; }

    virtual double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                      const Pythia8::PhaseSpace* phaseSpacePtr, bool inEvent)
    {
      //the variable selBias of the base class should be used;
      if ((sigmaProcessPtr->nFinal() == 2)) {
         selBias = 1.0/sigma.Eval(phaseSpacePtr->pTHat());
        return selBias;
      }
      selBias = 1.;
      return selBias;
    }

  private:
    double pTHatMin, pTHatMax;
    TF1 sigma;
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
