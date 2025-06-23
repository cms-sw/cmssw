#ifndef DaqSource_DaqFakeReader_h
#define DaqSource_DaqFakeReader_h

/** \class DaqFakeReader
 *  Generates empty FEDRawData of random size for all FEDs
 *  Proper headers and trailers are included; but the payloads are all 0s
 *  \author N. Amapane - CERN
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include <algorithm>

class DaqFakeReader : public edm::one::EDProducer<> {
public:
  DaqFakeReader(const edm::ParameterSet& pset);
  ~DaqFakeReader() override {}

  // Generate and fill FED raw data for a full event
  virtual int fillRawData(edm::Event& e, RawDataBuffer*& data);
  void produce(edm::Event&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void getSizes(const int, const int, float meansize, float width);
  void fillFEDs(const int, const int, edm::EventID& eID, RawDataBuffer& data, float meansize, float width);
  void fillTCDSFED(edm::EventID& eID, RawDataBuffer& data, uint32_t ls, timeval* now);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& iL, edm::EventSetup const& iE);

  edm::RunNumber_t runNum;
  edm::EventNumber_t eventNum;
  bool empty_events;
  bool fillRandom_;
  unsigned int meansize;  // in bytes
  unsigned int width;
  unsigned int injected_errors_per_million_events;
  unsigned int tcdsFEDID_;
  unsigned int modulo_error_events;
  unsigned int fakeLs_ = 0;
  std::vector<std::string> subsystems_;
  bool haveTCDS_ = false;
  bool haveSiPixel_ = false;
  bool haveSiStrip_ = false;
  bool haveECAL_ = false;
  bool haveHCAL_ = false;
  bool haveDT_ = false;
  bool haveCSC_ = false;
  bool haveRPC_ = false;

  uint32_t totSize_ = 0;
  std::vector<float> logSizes_;
  uint16_t logSizeIndex_;

};

#endif
