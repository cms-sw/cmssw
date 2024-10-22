#ifndef SHALLOW_EVENTDATA_PRODUCER
#define SHALLOW_EVENTDATA_PRODUCER

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include <vector>

class ShallowEventDataProducer : public edm::global::EDProducer<> {
public:
  explicit ShallowEventDataProducer(const edm::ParameterSet &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;

  edm::EDPutTokenT<unsigned int> runPut_;
  edm::EDPutTokenT<unsigned int> eventPut_;
  edm::EDPutTokenT<unsigned int> lumiPut_;
  edm::EDPutTokenT<unsigned int> bxPut_;
  edm::EDPutTokenT<float> instLumiPut_;
  edm::EDPutTokenT<float> puPut_;
#ifdef ExtendedCALIBTree
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> trig_token_;
  edm::EDPutTokenT<std::vector<bool>> trigTechPut_;
  edm::EDPutTokenT<std::vector<bool>> trigPhPut_;
#endif
};

#endif
