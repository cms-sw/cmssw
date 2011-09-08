#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"

HcalPulseContainmentManager::HcalPulseContainmentManager(
  float fixedphase_ns, float max_fracerror ) 
: entries_(),
  shapes_(),
  fixedphase_ns_(fixedphase_ns),
  max_fracerror_(max_fracerror)
{
}


void HcalPulseContainmentManager::beginRun(edm::EventSetup const & es)
{
  shapes_.beginRun(es);
}


void HcalPulseContainmentManager::endRun()
{
  shapes_.endRun();
}


double HcalPulseContainmentManager::correction(const HcalDetId & detId, 
                                               int toAdd, double fc_ampl)
{
  return get(detId, toAdd)->getCorrection(fc_ampl);
}


const HcalPulseContainmentCorrection * 
HcalPulseContainmentManager::get(const HcalDetId & detId, int toAdd)
{
  const HcalPulseShape * shape = &(shapes_.shape(detId));
  for(std::vector<HcalPulseContainmentEntry>::const_iterator entryItr = entries_.begin();
      entryItr != entries_.end(); ++entryItr)
  {
    if (entryItr->shape_ == shape && entryItr->toAdd_ == toAdd)
    {
        return &entryItr->correction_;
    }
  }
  // didn't find it.  Make one.
  HcalPulseContainmentEntry entry(toAdd, shape,
    HcalPulseContainmentCorrection(shape, toAdd, fixedphase_ns_, max_fracerror_));
  entries_.push_back(entry);
  return &(entries_.back().correction_);
} 

