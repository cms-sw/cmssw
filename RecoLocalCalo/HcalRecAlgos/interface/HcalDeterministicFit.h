#ifndef HcalDeterministicFit_h
#define HcalDeterministicFit_h 1

#include <typeinfo>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"


class HcalDeterministicFit {
 public:
  enum NegStrategy {DoNothing=0, ReqPos=1, FromGreg=2};  
  HcalDeterministicFit();
  ~HcalDeterministicFit();

  void init(HcalTimeSlew::ParaSource tsParam, HcalTimeSlew::BiasSetting bias, NegStrategy nStrat, PedestalSub pedSubFxn_);

  // This is the CMSSW Implementation of the apply function
  void apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & HLTOutput) const;
  // This is the edited implementation for our standalone test code
  void applyXM(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & HLTOutput) const;
  //void apply(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & HLTOutput) const;
  void getLandauFrac(float tStart, float tEnd, float &sum) const;

  //void SolveEquations(double *TS, double *par, double *fit) const;
  double det2(double *b, double *c) const;
  double det3(double *a, double *b, double *c) const;
  void pulseFraction(const double fC, double *TS46) const;

 private:
  HcalTimeSlew::ParaSource fTimeSlew;
  HcalTimeSlew::BiasSetting fTimeSlewBias;
  NegStrategy fNegStrat;
  PedestalSub fPedestalSubFxn_;
  
  static constexpr float landauFrac[] {0, 7.6377e-05, 0.000418655, 0.00153692, 0.00436844, 0.0102076, 
  0.0204177, 0.0360559, 0.057596, 0.0848493, 0.117069, 0.153152, 0.191858, 0.23198, 0.272461, 0.312438, 
  0.351262, 0.388476, 0.423788, 0.457036, 0.488159, 0.517167, 0.54412, 0.569112, 0.592254, 0.613668, 
  0.633402, 0.651391, 0.667242, 0.680131, 0.688868, 0.692188, 0.689122, 0.67928, 0.662924, 0.64087, 
  0.614282, 0.584457, 0.552651, 0.51997, 0.487317, 0.455378, 0.424647, 0.395445, 0.367963, 0.342288, 
  0.318433, 0.29636, 0.275994, 0.257243, 0.24, 0.224155, 0.2096, 0.196227, 0.183937, 0.172635, 
  0.162232, 0.15265, 0.143813, 0.135656, 0.128117, 0.12114, 0.114677, 0.108681, 0.103113, 0.0979354, 
  0.0931145, 0.0886206, 0.0844264, 0.0805074, 0.0768411, 0.0734075, 0.0701881, 0.0671664, 0.0643271, 
  0.0616564, 0.0591418, 0.0567718, 0.054536, 0.0524247, 0.0504292, 0.0485414, 0.046754, 0.0450602, 
  0.0434538, 0.041929, 0.0404806, 0.0391037, 0.0377937, 0.0365465, 0.0353583, 0.0342255, 0.0331447, 
  0.032113, 0.0311274, 0.0301854, 0.0292843, 0.0284221, 0.0275964, 0.0268053, 0.0253052, 0.0238536, 
  0.0224483, 0.0210872, 0.0197684, 0.0184899, 0.01725, 0.0160471, 0.0148795, 0.0137457, 0.0126445, 
  0.0115743, 0.0105341, 0.00952249, 0.00853844, 0.00758086, 0.00664871,0.00574103, 0.00485689, 0.00399541, 
  0.00315576, 0.00233713, 0.00153878, 0.000759962, 0 };
  
  
  //static double TS3par[3] = {0.44, -18.6, 5.136}; //Gaussian parameters: norm, mean, sigma for the TS3 fraction                    
  static constexpr double TS4par[] = {0.71, -5.17, 12.23}; //Gaussian parameters: norm, mean, sigma for the TS4 fraction                      
  static constexpr double TS5par[] = {0.258, 0.0178, 4.786e-4}; // pol2 parameters for the TS5 fraction                                       
  static constexpr double TS6par[] = {0.06391, 0.002737, 8.396e-05, 1.475e-06};// pol3 parameters for the TS6 fraction   
  
};

#endif // HLTAnalyzer_h
