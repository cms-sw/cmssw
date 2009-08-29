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

  // tfilterEnvelope stores doubles of energy and time; make sure we're checking over an even number of values
  // energies are also assumed to appear in increasing order

  // need at least two values to make comparison, and must
  // always have energy, time pair; otherwise, assume "in time" and don't set bits
  if (tfilterEnvelope_.size()<2 || tfilterEnvelope_.size()%2!=0)
      return;

  double twinmin, twinmax;
  double rhtime=hbhe.time();
  double energy=hbhe.energy();
  unsigned int i=0;

  if (energy<tfilterEnvelope_[0])
    twinmax=tfilterEnvelope_[1];
  else
    {
      for (i=0;i<2*(tfilterEnvelope_.size()/2);i+=2)
	{
	  // Identify tfilterEnvelope index for this rechit energy
	  if (tfilterEnvelope_[i]>energy)
	    break;
	}

      if (i==tfilterEnvelope_.size())
	twinmax=tfilterEnvelope_[i-1];
      else
	{
	  // Perform linear interpolation between energy boundaries

	  // i-2...i+1 are ensured to exist by our earlier requirement
	  // that envelope size be >=2 and even
	  double energy1  = tfilterEnvelope_[i-2];
	  double lim1     = tfilterEnvelope_[i-1];
	  double energy2  = tfilterEnvelope_[i];
	  double lim2     = tfilterEnvelope_[i+1];
	
	  twinmax=lim1+((lim2-lim1)*(energy-energy1)/(energy2-energy1));
	}
    }
  twinmax=0+twinmax*1;  // add offset and gain at some point?
  twinmin=0-twinmax*1;  // add offset and gain at some point?
   if (rhtime<=twinmin || rhtime >= twinmax)
    status=1; // set status to 1
   // Though we're only using a single bit right now, 3 bits are reserved for these cuts
  hbhe.setFlagField(status,HcalCaloFlagLabels::HBHETimingShapedCutsBits,3);
  return;
}
