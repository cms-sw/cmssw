#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMultiFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMultiFitAlgo_HH

/** \class EcalUncalibRecHitMultiFitAlgo
  *  Amplitude reconstucted by the multi-template fit
  *
  *  \author J.Bendavid, E.Di Marco
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"


#include "TMatrixDSym.h"
#include "TVectorD.h"

class EcalUncalibRecHitMultiFitAlgo
{
  
 public:
  
  EcalUncalibRecHitMultiFitAlgo();
  ~EcalUncalibRecHitMultiFitAlgo() { };
  EcalUncalibratedRecHit makeRecHit(const EcalDataFrame& dataFrame, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const SampleMatrix &noisecor, const FullSampleVector &fullpulse, const FullSampleMatrix &fullpulsecov, const BXVector &activeBX);
  void disableErrorCalculation() { _computeErrors = false; }
  
 private:
   PulseChiSqSNNLS _pulsefunc;
   bool _computeErrors;

};

#endif
