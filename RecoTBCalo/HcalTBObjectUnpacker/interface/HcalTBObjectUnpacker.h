#ifndef HcalTBObjectUnpacker_h
#define HcalTBObjectUnpacker_h

/** \class HcalTBObjectUnpacker
 *
 * HcalTBObjectUnpacker is the EDProducer subclass which runs 
 * the Hcal Test Beam Object Unpack algorithm.
 *
 * \author Phil Dudero
      
 *
 * \version   1st Version June 10, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTriggerDataUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBSlowDataUnpacker.h"
#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalTBTDCUnpacker.h"

namespace hcaltb
{
  class HcalTBObjectUnpacker : public edm::EDProducer
  {
  public:
    explicit HcalTBObjectUnpacker(const edm::ParameterSet& ps);
    virtual ~HcalTBObjectUnpacker();
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    int triggerFed_;
    int sdFed_;
    int tdcFed_;
    HcalTBTriggerDataUnpacker tdUnpacker_;
    HcalTBSlowDataUnpacker    sdUnpacker_;
    HcalTBTDCUnpacker         tdcUnpacker_;
  };
}

#endif
