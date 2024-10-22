//-------------------------------------------------
//
/**  \class DTuROSDigiToRaw
 *
 *   L1 DT uROS Raw-to-Digi
 *
 *
 *
 *   J. Troconiz  - UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTRawToDigi_DTuROSDigiToRaw_h
#define DTRawToDigi_DTuROSDigiToRaw_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTDigi/interface/DTuROSControlData.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <FWCore/Framework/interface/stream/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include <string>

class DTReadOutMapping;

class DTuROSDigiToRaw : public edm::stream::EDProducer<> {
public:
  /// Constructor
  DTuROSDigiToRaw(const edm::ParameterSet& pset);

  /// Destructor
  ~DTuROSDigiToRaw() override;

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e, const edm::EventSetup& c, FEDRawDataCollection& data);

private:
  unsigned int eventNum;

  edm::InputTag DTDigiInputTag_;

  bool debug_;

  int nfeds_;

  std::vector<int> feds_;

  int bslts[DOCESLOTS], dslts[DOCESLOTS];

  std::vector<int> wslts[DOCESLOTS];

  // Operations

  //process data

  void process(int DTuROSFED,
               edm::Handle<DTDigiCollection> digis,
               edm::ESHandle<DTReadOutMapping> mapping,
               FEDRawDataCollection& data);

  // utilities
  void clear();

  int theCRT(int ddu, int ros);

  int theSLT(int ddu, int ros, int rob);

  int theLNK(int ddu, int ros, int rob);

  edm::InputTag getDTDigiInputTag() { return DTDigiInputTag_; }

  edm::EDGetTokenT<DTDigiCollection> Raw_token;
  edm::ESGetToken<DTReadOutMapping, DTReadOutMappingRcd> mapping_token_;
};
#endif
