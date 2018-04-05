#ifndef EventFilter_RPCRawToDigi_RPCTwinMuxRawToDigi_h
#define EventFilter_RPCRawToDigi_RPCTwinMuxRawToDigi_h

#include <cstdint>
#include <vector>
#include <utility>
#include <set>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

namespace edm {
class ConfigurationDescriptions;
class Event;
class EventSetup;
class ParameterSet;
class Run;
} // namespace edm

class RPCTwinMuxRawToDigi
    : public edm::stream::EDProducer<>
{
public:
    RPCTwinMuxRawToDigi(edm::ParameterSet const & config);
    ~RPCTwinMuxRawToDigi() override;

    static void compute_crc_64bit(std::uint16_t & crc, std::uint64_t const & word);

    static void fillDescriptions(edm::ConfigurationDescriptions & descs);

    void beginRun(edm::Run const & run, edm::EventSetup const & setup) override;
    void produce(edm::Event & event, edm::EventSetup const & setup) override;

protected:
    bool processCDFHeaders(int fed
                           , std::uint64_t const * & word, std::uint64_t const * & word_end
                           , std::uint16_t & crc
                           , RPCAMCLinkCounters & counters) const;
    bool processCDFTrailers(int fed, unsigned int nwords
                            , std::uint64_t const * & word, std::uint64_t const * & word_end
                            , std::uint16_t & crc
                            , RPCAMCLinkCounters & counters) const;
    bool processBlock(int fed
                      , std::uint64_t const * & word, std::uint64_t const * word_end
                      , std::uint16_t & crc
                      , RPCAMCLinkCounters & counters
                      , std::set<std::pair<RPCDetId, RPCDigi> > & digis) const;
    bool processTwinMux(int fed, unsigned int amc_number, unsigned int size
                        , std::uint64_t const * & word, std::uint64_t const * word_end
                        , std::uint16_t & crc
                        , RPCAMCLinkCounters & counters
                        , std::set<std::pair<RPCDetId, RPCDigi> > & digis) const;
    void processRPCRecord(int fed, unsigned int amc_number
                          , unsigned int bx_counter
                          , rpctwinmux::RPCRecord const & record
                          , RPCAMCLinkCounters & counters
                          , std::set<std::pair<RPCDetId, RPCDigi> > & digis
                          , int bx_min, int bx_max
                          , unsigned int link, unsigned int link_max) const;
    void putRPCDigis(edm::Event & event
                     , std::set<std::pair<RPCDetId, RPCDigi> > const & digis);
    void putCounters(edm::Event & event
                     , std::unique_ptr<RPCAMCLinkCounters> counters);

protected:
    edm::EDGetTokenT<FEDRawDataCollection> raw_token_;

    bool calculate_crc_, fill_counters_;
    int bx_min_, bx_max_;

    edm::ESWatcher<RPCTwinMuxLinkMapRcd> es_tm_link_map_watcher_;
    std::vector<int> feds_;
    edm::ESHandle<RPCAMCLinkMap> es_tm_link_map_;
    edm::ESHandle<RPCLBLinkMap> es_lb_link_map_;
};

#endif // EventFilter_RPCRawToDigi_RPCTwinMuxRawToDigi_h
