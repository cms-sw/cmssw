#ifndef EventFilter_RPCRawToDigi_RPCDigiMerger_h
#define EventFilter_RPCRawToDigi_RPCDigiMerger_h

#include <cstdint>
#include <vector>
#include <utility>
#include <set>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

// #include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
// #include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
// #include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
// #include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
// #include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// #include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

namespace edm {
class ConfigurationDescriptions;
class Event;
class EventSetup;
class ParameterSet;
class Run;
} // namespace edm

class RPCDigiMerger
    : public edm::stream::EDProducer<>
{
public:
    RPCDigiMerger(edm::ParameterSet const & config);
    ~RPCDigiMerger() override;

    // static void compute_crc_64bit(std::uint16_t & crc, std::uint64_t const & word);

    static void fillDescriptions(edm::ConfigurationDescriptions & descs);

    void beginRun(edm::Run const & run, edm::EventSetup const & setup) override;
    void produce(edm::Event & event, edm::EventSetup const & setup) override;


protected:
    edm::EDGetTokenT<RPCDigiCollection> TwinMux_token_;
    edm::EDGetTokenT<RPCDigiCollection> OMTF_token_;
    edm::EDGetTokenT<RPCDigiCollection> CPPF_token_;

};

#endif // EventFilter_RPCRawToDigi_RPCDigiMerger_h
