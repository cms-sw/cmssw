#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimingShapedFlag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*
v1.0 by Jeff Temple
29 August 2009

This takes the timing envelope algorithms developed by 
Phil Dudero and uses them to apply a RecHit Flag.
*/

HBHETimingShapedFlagSetter::HBHETimingShapedFlagSetter()
{
}

HBHETimingShapedFlagSetter::HBHETimingShapedFlagSetter(std::vector<double> tfilterEnvelope)
{
  tfilterEnvelope_=tfilterEnvelope;
}

void HBHETimingShapedFlagSetter::SetTimingShapedFlags(HBHERecHit& hbhe)
{
  int status=0;  // 3 bits reserved;status can range from 0-7

  // tfilterEnvelope stores energy/time pairs; make sure we're checking over an even number of values
  // energies are also assumed to appear in increasing order
  for (unsigned int i=0;i<2*(tfilterEnvelope_.size()/2);i=i+2)
    {
      // Continue on if energy is greater than threshold value
      if (hbhe.energy()>tfilterEnvelope_[i]) continue;
      // rechit energy is now below threshold; compare its time to the threshold time
      if (hbhe.time()>tfilterEnvelope_[i+1])
	status=1;
      // right now, status can be 0 or 1, but 3 bits have been reserved for this flag
      hbhe.setFlagField(status,HcalCaloFlagLabels::HBHETimingShapedCutsBits,3);
      break;
    }
  return;
}
