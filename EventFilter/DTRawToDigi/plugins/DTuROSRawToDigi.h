//-------------------------------------------------
//
/**  \class DTuROSRawToDigi
 *
 *   L1 DT uROS Raw-to-Digi
 *
 *
 *
 *   C. Heidemann - RWTH Aachen
 *   J. Troconiz  - UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTRawToDigi_DTuROSRawToDigi_h
#define DTRawToDigi_DTuROSRawToDigi_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <FWCore/Framework/interface/stream/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include <string>

class DTReadOutMapping;
class DTuROSFEDData;

class DTuROSRawToDigi : public edm::stream::EDProducer<> {
public:
  /// Constructor
  DTuROSRawToDigi(const edm::ParameterSet& pset);

  /// Destructor
  ~DTuROSRawToDigi() override;

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e, const edm::EventSetup& c, DTDigiCollection& digis, std::vector<DTuROSFEDData>& words);

private:
  edm::InputTag DTuROSInputTag_;

  bool debug_;

  int nfeds_;

  std::vector<int> feds_;

  unsigned char* lineFED;

  // Operations

  //process data

  void process(int DTuROSFED,
               edm::Handle<FEDRawDataCollection> data,
               edm::ESHandle<DTReadOutMapping> mapping,
               DTDigiCollection& digis,
               DTuROSFEDData& fwords);

  // utilities
  inline void readline(int& lines, long& dataWord) {
    dataWord = *((long*)lineFED);
    lineFED += 8;
    ++lines;
  }

  int theDDU(int crate, int slot, int link, bool tenDDU);

  int theROS(int slot, int link);

  int theROB(int slot, int link);

  edm::InputTag getDTuROSInputTag() { return DTuROSInputTag_; }

  edm::EDGetTokenT<FEDRawDataCollection> Raw_token;
  edm::ESGetToken<DTReadOutMapping, DTReadOutMappingRcd> mapping_token_;
};
#endif
