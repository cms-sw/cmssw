#ifndef RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1AlgoDebug_h_
#define RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1AlgoDebug_h_

#include <memory>
#include <vector>

// Base class header
#include "RecoLocalCalo/HcalRecAlgos/interface/SimpleHBHEPhase1Algo.h"

// Other headers
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/MahiFit.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

class SimpleHBHEPhase1AlgoDebug : public SimpleHBHEPhase1Algo 
{
 public:
  SimpleHBHEPhase1AlgoDebug(int firstSampleShift,
			    int samplesToAdd,
			    float phaseNS,
			    float timeShift,
			    bool correctForPhaseContainment,
			    std::unique_ptr<PulseShapeFitOOTPileupCorrection> m2,
			    std::unique_ptr<HcalDeterministicFit> detFit,
			    std::unique_ptr<MahiFit> mahi);
  inline ~SimpleHBHEPhase1AlgoDebug() override {}

  MahiDebugInfo recoDebug(const HBHEChannelInfo& info,
			  const bool isRealData,
			  const HcalTimeSlew *hcalTimeSlewDelay);
};

#endif // RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1AlgoDebug_h_
