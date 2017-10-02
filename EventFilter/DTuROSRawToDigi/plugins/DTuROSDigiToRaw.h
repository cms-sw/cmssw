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
#ifndef DTuROSRawToDigi_DTuROSDigiToRaw_h
#define DTuROSRawToDigi_DTuROSDigiToRaw_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include <string>

class DTReadOutMapping;


class DTuROSDigiToRaw : public edm::EDProducer {

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

  unsigned char* LineFED;

  int bslts[13], dslts[13];

  std::vector<int> wslts[13];

  // Operations

  //process data

  void process(int DTuROSFED,
               edm::Handle<DTDigiCollection> digis,
               edm::ESHandle<DTReadOutMapping> mapping,
               FEDRawDataCollection& data);

  // utilities
  void clear();

  void calcCRC(int myD1, int myD2, int& myC);

  int theCRT(int ddu, int ros, int rob);

  int theSLT(int ddu, int ros, int rob);

  int theLNK(int ddu, int ros, int rob);

  edm::InputTag getDTDigiInputTag() { return DTDigiInputTag_; }
  
  edm::EDGetTokenT<DTDigiCollection> Raw_token;

};
#endif
