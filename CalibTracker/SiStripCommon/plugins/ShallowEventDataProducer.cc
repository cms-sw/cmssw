#include "ShallowEventDataProducer.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include <bitset>

ShallowEventDataProducer::ShallowEventDataProducer(const edm::ParameterSet& iConfig) {
  runPut_ = produces<unsigned int>("run");
  eventPut_ = produces<unsigned int>("event");
  bxPut_ = produces<unsigned int>("bx");
  lumiPut_ = produces<unsigned int>("lumi");
  instLumiPut_ = produces<float>("instLumi");
  puPut_ = produces<float>("PU");
#ifdef ExtendedCALIBTree
  trigTechPut_ = produces<std::vector<bool>>("TrigTech");
  trigPhPut_ = produces<std::vector<bool>>("TrigPh");
  trig_token_ = consumes<L1GlobalTriggerReadoutRecord>(iConfig.getParameter<edm::InputTag>("trigRecord"));
#endif

  scalerToken_ = consumes<LumiScalersCollection>(iConfig.getParameter<edm::InputTag>("lumiScalers"));
  metaDataToken_ = consumes<OnlineLuminosityRecord>(iConfig.getParameter<edm::InputTag>("metadata"));
}

void ShallowEventDataProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.emplace(runPut_, iEvent.id().run());
  iEvent.emplace(eventPut_, iEvent.id().event());
  iEvent.emplace(bxPut_, iEvent.bunchCrossing());
  iEvent.emplace(lumiPut_, iEvent.luminosityBlock());

#ifdef ExtendedCALIBTree
  edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  iEvent.getByToken(trig_token_, gtRecord);

  std::vector<bool> TrigTech_(64, false);
  std::vector<bool> TrigPh_(128, false);

  // Get dWord after masking disabled bits
  DecisionWord dWord = gtRecord->decisionWord();
  if (!dWord.empty()) {  // if board not there this is zero
    // loop over dec. bit to get total rate (no overlap)
    for (int i = 0; i < 64; ++i) {
      TrigPh_[i] = dWord[i];
    }
  }

  TechnicalTriggerWord tw = gtRecord->technicalTriggerWord();
  if (!tw.empty()) {
    // loop over dec. bit to get total rate (no overlap)
    for (int i = 0; i < 64; ++i) {
      TrigTech_[i] = tw[i];
    }
  }

  iEvent.emplace(trigTechPut_, std::move(TrigTech_));
  iEvent.emplace(trigPhPut_, std::move(TrigPh_));
#endif

  // Luminosity informations
  edm::Handle<LumiScalersCollection> lumiScalers = iEvent.getHandle(scalerToken_);
  edm::Handle<OnlineLuminosityRecord> metaData = iEvent.getHandle(metaDataToken_);

  float instLumi_ = 0;
  float PU_ = 0;

  if (lumiScalers.isValid() && !lumiScalers->empty()) {
    if (lumiScalers->begin() != lumiScalers->end()) {
      instLumi_ = lumiScalers->begin()->instantLumi();
      PU_ = lumiScalers->begin()->pileup();
    }
  } else if (metaData.isValid()) {
    instLumi_ = metaData->instLumi();
    PU_ = metaData->avgPileUp();
  } else {
    edm::LogInfo("ShallowEventDataProducer")
        << "LumiScalers collection not found in the event; will write dummy values";
  }

  iEvent.emplace(instLumiPut_, instLumi_);
  iEvent.emplace(puPut_, PU_);
}
