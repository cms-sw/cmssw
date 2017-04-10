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

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/HcalRawToDigi/interface/HcalPacker.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class HcalDigiToRaw : public edm::global::EDProducer<>
{
public:
  explicit HcalDigiToRaw(const edm::ParameterSet& ps);
  virtual ~HcalDigiToRaw();
  virtual void produce(edm::StreamID id, edm::Event& e, const edm::EventSetup& c) const override;
private:
  HcalPacker packer_;
  const edm::InputTag hbheTag_, hoTag_, hfTag_, zdcTag_, calibTag_, trigTag_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  edm::EDGetTokenT<ZDCDigiCollection> tok_zdc_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_htp_;
};

#endif
