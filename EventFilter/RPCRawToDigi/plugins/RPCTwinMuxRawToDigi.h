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
    RPCTwinMuxRawToDigi(edm::ParameterSet const & _config);
    ~RPCTwinMuxRawToDigi();

    static void compute_crc_64bit(std::uint16_t & _crc, std::uint64_t const & _word);

    static void fillDescriptions(edm::ConfigurationDescriptions & _descs);

    void beginRun(edm::Run const & _run, edm::EventSetup const & _setup) override;
    void produce(edm::Event & _event, edm::EventSetup const & _setup) override;

protected:
    bool processCDFHeaders(int _fed
                           , std::uint64_t const * & _word, std::uint64_t const * & _word_end
                           , std::uint16_t & _crc
                           , RPCAMCLinkCounters & _counters) const;
    bool processCDFTrailers(int _fed, unsigned int _nwords
                            , std::uint64_t const * & _word, std::uint64_t const * & _word_end
                            , std::uint16_t & _crc
                            , RPCAMCLinkCounters & _counters) const;
    bool processBlock(int _fed
                      , std::uint64_t const * & _word, std::uint64_t const * _word_end
                      , std::uint16_t & _crc
                      , RPCAMCLinkCounters & _counters
                      , std::set<std::pair<RPCDetId, RPCDigi> > & _digis) const;
    bool processTwinMux(int _fed, unsigned int _amc_number, unsigned int _size
                        , std::uint64_t const * & _word, std::uint64_t const * _word_end
                        , std::uint16_t & _crc
                        , RPCAMCLinkCounters & _counters
                        , std::set<std::pair<RPCDetId, RPCDigi> > & _digis) const;
    void processRPCRecord(int _fed, unsigned int _amc_number
                          , unsigned int _bx_counter
                          , rpctwinmux::RPCRecord const & _record
                          , RPCAMCLinkCounters & _counters
                          , std::set<std::pair<RPCDetId, RPCDigi> > & _digis
                          , int _bx_min, int _bx_max
                          , unsigned int _link, unsigned int _link_max) const;
    void putRPCDigis(edm::Event & _event
                     , std::set<std::pair<RPCDetId, RPCDigi> > const & _digis);
    void putCounters(edm::Event & _event
                     , std::unique_ptr<RPCAMCLinkCounters> _counters);

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
