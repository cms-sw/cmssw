#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRecordProducer_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRecordProducer_h

/**
 * \class L1GlobalTriggerRecordProducer
 * 
 * 
 * Description: L1GlobalTriggerRecord producer.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna 
 * 
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

// forward declarations
class L1GtTriggerMask;

// class declaration
class L1GlobalTriggerRecordProducer : public edm::stream::EDProducer<> {
public:
  /// constructor(s)
  explicit L1GlobalTriggerRecordProducer(const edm::ParameterSet&);

  /// destructor
  ~L1GlobalTriggerRecordProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// cached stuff

  /// trigger masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmTech;
  unsigned long long m_l1GtTmTechCacheID;

  std::vector<unsigned int> m_triggerMaskAlgoTrig;
  std::vector<unsigned int> m_triggerMaskTechTrig;

private:
  /// InputTag for the L1 Global Trigger DAQ readout record
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtReadoutRecordTag;
  /// EventSetup Token for l1GtTriggerMaskAlgo
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_l1GtTmAlgoToken;
  ///  EventSetupToken for L1GtTriggerMaskTech
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_l1GtTmTechToken;
};

#endif  // EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRecordProducer_h
