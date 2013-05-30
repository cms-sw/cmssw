#include <iostream>
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

HBHETimingShapedFlagSetter::HBHETimingShapedFlagSetter(const std::vector<double>& v_userEnvelope)
{
  makeTfilterEnvelope(v_userEnvelope);
  
  ignorelowest_=false;
  ignorehighest_=false;
  win_offset_=0.;
  win_gain_=1.;
}

HBHETimingShapedFlagSetter::HBHETimingShapedFlagSetter(const std::vector<double>& v_userEnvelope,
						       bool ignorelowest, bool ignorehighest,
						       double win_offset, double win_gain)
{
  makeTfilterEnvelope(v_userEnvelope);

  ignorelowest_  = ignorelowest;  // can ignore flagging hits below lowest energy threshold
  ignorehighest_ = ignorehighest; // can ignore flagging hits above highest energy threshold
  win_offset_    = win_offset;    // timing offset
  win_gain_      = win_gain;      // time gain
}

void
HBHETimingShapedFlagSetter::makeTfilterEnvelope(const std::vector<double>& v_userEnvelope)
{
  // Transform vector of doubles into a map of <energy,lowtime,hitime> triplets
  // Add extra protection in case vector of doubles is not a multiple of 3
  if (v_userEnvelope.size()%3)
    throw cms::Exception("Invalid tfilterEnvelope definition") <<
      "Must be one energy and two times per point";

  for (unsigned int i=0;i<v_userEnvelope.size();i+=3) {
    int intGeV = (int)(v_userEnvelope[i]+0.5);
    std::pair<double,double> pairOfTimes = std::make_pair(v_userEnvelope[i+1],
							  v_userEnvelope[i+2]);
    if ((pairOfTimes.first  > 0) ||
	(pairOfTimes.second < 0) )
      throw cms::Exception("Invalid tfilterEnvelope definition") <<
	"Min and max time values must straddle t=0; use win_offset to shift";

    tfilterEnvelope_[intGeV] = pairOfTimes;
  }
}

void
HBHETimingShapedFlagSetter::dumpInfo()
{  
  std::cout <<"Timing Energy/Time parameters are:"<<std::endl;
  TfilterEnvelope_t::const_iterator it;
  for (it=tfilterEnvelope_.begin();it!=tfilterEnvelope_.end();++it)
    std::cout <<"\t"<<it->first<<" GeV\t"<<it->second.first<<" ns\t"<<it->second.second<<" ns"<<std::endl;

  std::cout <<"ignorelowest  = "<<ignorelowest_<<std::endl;
  std::cout <<"ignorehighest = "<<ignorehighest_<<std::endl;
  std::cout <<"win_offset    = "<<win_offset_<<std::endl;
  std::cout <<"win_gain      = "<<win_gain_<<std::endl;
}

int
HBHETimingShapedFlagSetter::timingStatus(const HBHERecHit& hbhe)
{
  int status=0;  // 3 bits reserved;status can range from 0-7

  // tfilterEnvelope stores triples of energy and high/low time limits; 
  // make sure we're checking over an even number of values
  // energies are guaranteed by std::map to appear in increasing order

  // need at least two values to make comparison, and must
  // always have energy, time pair; otherwise, assume "in time" and don't set bits
  if (tfilterEnvelope_.size()==0)
    return 0;

  double twinmin, twinmax;  // min, max 'good' time; values outside this range have flag set
  double rhtime=hbhe.time();
  double energy=hbhe.energy();

  if (energy< (double)tfilterEnvelope_.begin()->first) // less than lowest energy threshold
    {
      // Can skip timing test on cells below lowest threshold if so desired
      if (ignorelowest_) 
	return 0;
      else {
	twinmin=tfilterEnvelope_.begin()->second.first;
	twinmax=tfilterEnvelope_.begin()->second.second;
      }
    }
  else
    {
      // Loop over energies in tfilterEnvelope
      TfilterEnvelope_t::const_iterator it;
      for (it=tfilterEnvelope_.begin();it!=tfilterEnvelope_.end();++it)
	{
	  // Identify tfilterEnvelope index for this rechit energy
	  if (energy < (double)it->first)
	    break;
	}

      if (it==tfilterEnvelope_.end())
	{
	  // Skip timing test on cells above highest threshold if so desired
	  if (ignorehighest_)
	    return 0;
	  else {
	    twinmin=tfilterEnvelope_.rbegin()->second.first;
	    twinmax=tfilterEnvelope_.rbegin()->second.second;
	  }
	}
      else
	{
	  // Perform linear interpolation between energy boundaries

	  std::map<int,std::pair<double,double> >::const_iterator prev = it; prev--;

	  // twinmax interpolation
	  double energy1  = prev->first;
	  double lim1     = prev->second.second;
	  double energy2  = it->first;
	  double lim2     = it->second.second;
	
	  twinmax=lim1+((lim2-lim1)*(energy-energy1)/(energy2-energy1));

	  // twinmin interpolation
	  lim1     = prev->second.first;
	  lim2     = it->second.first;
	
	  twinmin=lim1+((lim2-lim1)*(energy-energy1)/(energy2-energy1));
	}
    }

  // Apply offset, gain
  twinmin=win_offset_+twinmin*win_gain_;
  twinmax=win_offset_+twinmax*win_gain_;  

  // Set status high if time outside expected range
  if (rhtime<=twinmin || rhtime >= twinmax)
    status=1; // set status to 1

  return status;
}

void HBHETimingShapedFlagSetter::SetTimingShapedFlags(HBHERecHit& hbhe)
{
  int status = timingStatus(hbhe);

  // Though we're only using a single bit right now, 3 bits are reserved for these cuts
  hbhe.setFlagField(status,HcalCaloFlagLabels::HBHETimingShapedCutsBits,3);

  return;
}
