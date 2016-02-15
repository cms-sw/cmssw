#ifndef HcalUTCAhistogramUnpacker_h
#define HcalUTCAhistogramUnpacker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoTBCalo/HcalTBObjectUnpacker/interface/HcalSourcingUTCAunpacker.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

  class HcalUTCAhistogramUnpacker : public edm::EDProducer
  {
  public:
    explicit HcalUTCAhistogramUnpacker(const edm::ParameterSet& ps);
    virtual ~HcalUTCAhistogramUnpacker();
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  private:
    HcalSourcingUTCAunpacker histoUnpacker_;
    edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
    std::string electronicsMapLabel_;
  };


#endif
