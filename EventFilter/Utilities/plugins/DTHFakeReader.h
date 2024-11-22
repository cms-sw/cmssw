#ifndef DaqSource_DTHFakeReader_h
#define DaqSource_DTHFakeReader_h

/** \class DTHFakeReader
 *  Fills FedRawData with DTH orbits for writeout to emulate EVB file writing
 *  Proper Phase-2 headers and trailers are included;
 */

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Utilities/interface/DTHHeaders.h"
#include <algorithm>

namespace evf {

  class DTHFakeReader : public edm::one::EDProducer<> {
  public:
    DTHFakeReader(const edm::ParameterSet& pset);
    ~DTHFakeReader() override {}

    void produce(edm::Event&, edm::EventSetup const&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // Generate and fill FED raw data for a full event
    void fillRawData(edm::Event& e, FEDRawDataCollection*& data);

    uint32_t fillSLRFED(unsigned char* buf, const uint32_t sourceId, edm::EventNumber_t eventId, const uint32_t orbitId, uint32_t size, uint32_t &accum_crc32c);
    uint32_t fillFED(unsigned char* buf, const int sourceId, edm::EventNumber_t eventId, uint32_t size, uint32_t &accum_crc32c);
    //void fillTCDSFED(edm::EventID& eID, FEDRawDataCollection& data, uint32_t ls, timeval* now);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& iL, edm::EventSetup const& iE);

  private:
    bool fillRandom_;
    unsigned int meansize_;  // in bytes
    unsigned int width_;
    unsigned int injected_errors_per_million_events_;
    std::vector<unsigned int> sourceIdList_;
    unsigned int modulo_error_events_;
    unsigned int fakeLs_ = 0;
  };
} //namespace evf

#endif
