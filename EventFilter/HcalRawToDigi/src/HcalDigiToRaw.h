#ifndef HcalDigiToRaw_h
#define HcalDigiToRaw_h

/** \class HcalDigiToRaw
 *
 * HcalDigiToRaw is the EDProducer subclass which runs 
 * the Hcal Unpack algorithm.
 *
 * \author Jeremiah Mans
      
 *
 * \version   1st Version June 10, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/HcalRawToDigi/interface/HcalPacker.h"

class HcalDigiToRaw : public edm::EDProducer
{
public:
  explicit HcalDigiToRaw(const edm::ParameterSet& ps);
  virtual ~HcalDigiToRaw();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  HcalPacker packer_;
  edm::InputTag hbheTag_, hoTag_, hfTag_, zdcTag_, calibTag_, trigTag_;
};

#endif
