#ifndef CSCDCCUnpacker_h
#define CSCDCCUnpacker_h

/** \class CSCDCCUnpacker
 * 
 *
 * \author Alex Tumanov 
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"

class CSCMonitorInterface;

class CSCDCCUnpacker : public edm::stream::EDProducer<> {
public:
  /// Constructor
  CSCDCCUnpacker(const edm::ParameterSet& pset);

  /// Destructor
  ~CSCDCCUnpacker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  /// Visualization of raw data in FED-less events (Robert Harr and Alexander Sakharov)
  void visual_raw(int hl, int id, int run, int event, bool fedshort, bool fDump, short unsigned int* buf) const;

private:
  bool debug, printEventNumber, goodEvent, useExaminer, unpackStatusDigis;
  bool useSelectiveUnpacking, useFormatStatus;

  /// option to unpack GEM cluster data
  bool useGEMs_;

  /// Visualization of raw data
  bool visualFEDInspect, visualFEDShort, formatedEventDump;
  /// Suppress zeros LCTs
  bool SuppressZeroLCT;

  int numOfEvents;
  unsigned int errorMask, examinerMask;
  bool instantiateDQM;
  CSCMonitorInterface* monitor;

  /// Token for consumes interface & access to data
  edm::EDGetTokenT<FEDRawDataCollection> i_token;
  edm::ESGetToken<CSCCrateMap, CSCCrateMapRcd> crateToken;
  edm::ESGetToken<CSCChamberMap, CSCChamberMapRcd> cscmapToken;
};

#endif
