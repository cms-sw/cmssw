#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include <iostream>

HcalPulseContainmentManager::HcalPulseContainmentManager(float max_fracerror ) 
: entries_(),
  shapes_(),
  max_fracerror_(max_fracerror)
{
  hcalTimeSlew_delay_ = nullptr;
}

void HcalPulseContainmentManager::beginRun(edm::EventSetup const & es)
{
  edm::ESHandle<HcalTimeSlew> delay;
  es.get<HcalTimeSlewRecord>().get("HBHE", delay);
  hcalTimeSlew_delay_ = &*delay;

  shapes_.beginRun(es);
}

void HcalPulseContainmentManager::beginRun(const HcalDbService* conditions, const HcalTimeSlew* delay)
{
  hcalTimeSlew_delay_ = delay;

  shapes_.beginRun(conditions);
}

double HcalPulseContainmentManager::correction(const HcalDetId & detId, 
                                               int toAdd, float fixedphase_ns, double fc_ampl)
{

  /*
          int sub     = detId.subdet();
          int depth   = detId.depth();
          int inteta  = detId.ieta();
          int intphi  = detId.iphi();
          
          std::cout << "* HcalPulseContainmentManager::correction,   cell:" 
                    << " sub, ieta, iphi, depth = " 
                    << sub << "  " << inteta << "  " << intphi 
                    << "  " << depth  << "  toAaa= " << toAdd 
		    <<"  phase = " << fixedphase_ns << "  ampl = " 
		    << fc_ampl 
                    << std::endl;
  */

  return get(detId, toAdd, fixedphase_ns)->getCorrection(fc_ampl);
}

const HcalPulseContainmentCorrection * 
HcalPulseContainmentManager::get(const HcalDetId & detId, int toAdd, float fixedphase_ns)
{
  // const HcalPulseShape * shape = &(shapes_.shape(detId));
  const HcalPulseShape * shape = &(shapes_.shapeForReco(detId));
  for(std::vector<HcalPulseContainmentEntry>::const_iterator entryItr = entries_.begin();
      entryItr != entries_.end(); ++entryItr)
  {
    if (entryItr->shape_ == shape && entryItr->toAdd_ == toAdd && entryItr->fixedphase_ns_ == fixedphase_ns)
    {
        return &entryItr->correction_;
    }
  }

  /*
          int sub     = detId.subdet();
          int depth   = detId.depth();
          int inteta  = detId.ieta();
          int intphi  = detId.iphi();
          
          std::cout << "* HcalPulseContainmentManager::get new entry,  cell:" 
                    << " sub, ieta, iphi, depth = " 
                    << sub << "  " << inteta << "  " << intphi 
                    << "  " << depth   
                    << std::endl;
  */

  // didn't find it.  Make one.
  HcalPulseContainmentEntry entry(toAdd, fixedphase_ns, shape, HcalPulseContainmentCorrection(shape, toAdd, fixedphase_ns, max_fracerror_, hcalTimeSlew_delay_));
  entries_.push_back(entry);
  return &(entries_.back().correction_);
}
