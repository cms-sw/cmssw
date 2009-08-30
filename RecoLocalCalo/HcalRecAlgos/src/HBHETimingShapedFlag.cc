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
  // Transform vector of doubles into a vector of <energy,time> pairs
  // Add extra protection in case vector of doubles is not a multiple of 2?
  if (tfilterEnvelope.size()%2==0)
    {
      for (unsigned int i=0;i<tfilterEnvelope.size();i+=2)
	tfilterEnvelope_.push_back(std::pair<double,double>(tfilterEnvelope[i],
							    tfilterEnvelope[i+1]));
    }
  
  //sort in order of increasing energy -- unnecessary?
  std::sort(tfilterEnvelope_.begin(),
	    tfilterEnvelope_.end(),
	    compareEnergyTimePair<std::pair<double,double> >());

  ignorelowest_=false;
  ignorehighest_=false;
  win_offset_=0.;
  win_gain_=1.;
}

HBHETimingShapedFlagSetter::HBHETimingShapedFlagSetter(std::vector<double> tfilterEnvelope, bool ignorelowest, bool ignorehighest, double win_offset, double win_gain)
{
  // Transform vector of doubles into a vector of <energy,time> pairs
  // Add extra protection in case vector of doubles is not a multiple of 2?
  if (tfilterEnvelope.size()%2==0)
    {
      for (unsigned int i=0;i<tfilterEnvelope.size();i+=2)
	tfilterEnvelope_.push_back(std::pair<double,double>(tfilterEnvelope[i],
							    tfilterEnvelope[i+1]));
    }

  //sort in order of increasing energy -- unnecessary?
  std::sort(tfilterEnvelope_.begin(),
	    tfilterEnvelope_.end(),
	    compareEnergyTimePair<std::pair<double,double> >());

  ignorelowest_=ignorelowest; // can ignore flagging hits below lowest energy threshold
  ignorehighest_=ignorehighest; // can ignore flagging hits above highest energy threshold
  win_offset_=win_offset; // timing offset
  win_gain_=win_gain;  // time gain
}

void HBHETimingShapedFlagSetter::SetTimingShapedFlags(HBHERecHit& hbhe)
{
  int status=0;  // 3 bits reserved;status can range from 0-7

  // tfilterEnvelope stores doubles of energy and time; 
  //make sure we're checking over an even number of values
  // energies are also assumed to appear in increasing order

  // need at least two values to make comparison, and must
  // always have energy, time pair; otherwise, assume "in time" and don't set bits
  if (tfilterEnvelope_.size()==0)
    return;

  double twinmin, twinmax;  // min, max 'good' time; values outside this range have flag set
  double rhtime=hbhe.time();
  double energy=hbhe.energy();
  unsigned int i=0; // index to track envelope index

  if (energy<tfilterEnvelope_[0].first) // less than lowest energy threshold
    {
      // Can skip timing test on cells below lowest threshold if so desired
      if (ignorelowest_) 
	return;
      else
	twinmax=tfilterEnvelope_[0].second;
    }
  else
    {
      // Loop over energies in tfilterEnvelope
      for (i=0;i<tfilterEnvelope_.size();++i)
	{
	  // Identify tfilterEnvelope index for this rechit energy
	  if (tfilterEnvelope_[i].first>energy)
	    break;
	}

      if (i==tfilterEnvelope_.size())
	{
	  // Skip timing test on cells above highest threshold if so desired
	  if (ignorehighest_)
	    return;
	  else
	    twinmax=tfilterEnvelope_[i-1].second;
	}
      else
	{
	  // Perform linear interpolation between energy boundaries

	  double energy1  = tfilterEnvelope_[i-1].first;
	  double lim1     = tfilterEnvelope_[i-1].second;
	  double energy2  = tfilterEnvelope_[i].first;
	  double lim2     = tfilterEnvelope_[i].second;
	
	  twinmax=lim1+((lim2-lim1)*(energy-energy1)/(energy2-energy1));
	}
    }
  // Apply offset, gain
  twinmax=win_offset_+twinmax*win_gain_;  
  twinmin=win_offset_-twinmax*win_gain_;
  // Set status high if time outside expected range
  if (rhtime<=twinmin || rhtime >= twinmax)
    status=1; // set status to 1

   // Though we're only using a single bit right now, 3 bits are reserved for these cuts
  hbhe.setFlagField(status,HcalCaloFlagLabels::HBHETimingShapedCutsBits,3);
  return;
}
