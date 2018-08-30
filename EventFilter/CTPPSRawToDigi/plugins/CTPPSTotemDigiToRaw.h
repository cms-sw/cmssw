#ifndef CTPPSTotemDigiToRaw_H
#define CTPPSTotemDigiToRaw_H

/** \class CTPPSTotemDigiToRaw_H
 */

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemFramePosition.h"
#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"

class CTPPSTotemDigiToRaw final : public edm::EDProducer {
public:

  /// ctor
  explicit CTPPSTotemDigiToRaw( const edm::ParameterSet& );

  /// dtor
  ~CTPPSTotemDigiToRaw() override;


  /// dummy end of job 
  void endJob() override {}

  /// get data, convert to raw event, attach again to Event
  void produce( edm::Event&, const edm::EventSetup& ) override;
  typedef uint64_t Word64;

private:
  edm::ParameterSet config_;
  unsigned long eventCounter;
  std::set<unsigned int> fedIds_;
  edm::InputTag label_;  //label of input digi data
  int allDigiCounter;
  int allWordCounter;
  edm::ESWatcher<TotemReadoutRcd> recordWatcher;
  bool debug;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPDigi>> tTotemRPDigi; 
  std::map<std::map<const uint32_t, unsigned int>, std::map<short unsigned int, std::map<short unsigned int, short unsigned int>>> iDdet2fed_;
  TotemFramePosition fPos_;

};
#endif
