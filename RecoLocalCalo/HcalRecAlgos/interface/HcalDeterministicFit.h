#ifndef RecoLocalCalo_HcalRecAlgos_HcalDeterministicFit_h
#define RecoLocalCalo_HcalRecAlgos_HcalDeterministicFit_h

#include <typeinfo>
#include <vector>

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PedestalSub.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

class HcalDeterministicFit {
 public:
  HcalDeterministicFit();
  ~HcalDeterministicFit();

  void init(HcalTimeSlew::ParaSource tsParam, HcalTimeSlew::BiasSetting bias, bool iApplyTimeSlew, PedestalSub pedSubFxn_, std::vector<double> pars, double respCorr);

  void phase1Apply(const HBHEChannelInfo& channelData,
		   float& reconstructedEnergy,
		   float& reconstructedTime) const;

  // This is the CMSSW Implementation of the apply function
  void getLandauFrac(float tStart, float tEnd, float &sum) const;
  void get205Frac(float tStart, float tEnd, float &sum) const;

 private:
  HcalTimeSlew::ParaSource fTimeSlew;
  HcalTimeSlew::BiasSetting fTimeSlewBias;
  PedestalSub fPedestalSubFxn_;
  bool applyTimeSlew_;

  double fpars[9];
  double frespCorr;
 
  static constexpr int HcalRegion[2] = {16, 17};
  static constexpr int tsWidth = 25;
  static constexpr float negThresh[2] = {-3., 15.};
  static constexpr float invGpar[3] = {-13.11, 11.29, 5.133};
  static constexpr float rCorr[2] = {0.95, 0.95};
  static constexpr float rCorrSiPM[2] = {1., 1.};

  static constexpr float landauFrac[125] = { 0, 7.6377e-05, 0.000418655, 0.00153692, 0.00436844, 0.0102076, 
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

  static constexpr float siPM205Frac[125] = {
    0, 0, 0, 0, 0.00133129, 0.00444633, 0.0115, 0.0243992, 0.0443875, 0.0716386, 0.105298, 0.143832,
    0.185449, 0.228439, 0.271367, 0.31315, 0.353041, 0.390587, 0.425555, 0.45788, 0.487604, 0.514843,
    0.539752, 0.562504, 0.583282, 0.602263, 0.619612, 0.635457, 0.649765, 0.66208, 0.671249, 0.675509,
    0.673048, 0.662709, 0.644394, 0.619024, 0.588194, 0.55375, 0.517448, 0.480768, 0.444831, 0.410418,
    0.378015, 0.347879, 0.320103, 0.294667, 0.271474, 0.250391, 0.231257, 0.213907, 0.198178, 0.183914,
    0.170967, 0.159205, 0.148505, 0.138758, 0.129864, 0.121737, 0.114299, 0.107478, 0.101214, 0.0954507,
    0.0901402, 0.0852385, 0.0807069, 0.0765108, 0.0726194, 0.0690052, 0.0656435, 0.0625123, 0.0595916, 0.0568637,
    0.0543125, 0.0519236, 0.0496838, 0.0475815, 0.0456058, 0.0437472, 0.0419966, 0.0403463, 0.0387887, 0.0373173,
    0.0359259, 0.034609, 0.0333615, 0.0321786, 0.0310561, 0.02999, 0.0289767, 0.0280127, 0.0270951, 0.0262209,
    0.0253875, 0.0245923, 0.0238333, 0.0231082, 0.022415, 0.021752, 0.0211174, 0.0205097, 0.0199274, 0.0193692,
    0.0188336, 0.0183196, 0.017826, 0.0173518, 0.0168959, 0.0164575, 0.0160356, 0.0156296, 0.0152385, 0.0148617,
    0.0144984, 0.0141482, 0.0138103, 0.0134842, 0.0131693, 0.0128652, 0.0125714, 0.0122873, 0.0120127, 0.011747,
    0.01149, 0.0112412, 0.0110002 };

};

#endif // HLTAnalyzer_h
