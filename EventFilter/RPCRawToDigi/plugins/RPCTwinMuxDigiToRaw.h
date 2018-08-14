#ifndef EventFilter_RPCRawToDigi_RPCTwinMuxDigiToRaw_h
#define EventFilter_RPCRawToDigi_RPCTwinMuxDigiToRaw_h

#include <map>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

namespace edm {
class ConfigurationDescriptions;
class Event;
class EventSetup;
class ParameterSet;
class Run;
} // namespace edm

class RPCTwinMuxDigiToRaw
    : public edm::stream::EDProducer<>
{
public:
    RPCTwinMuxDigiToRaw(edm::ParameterSet const & config);
    ~RPCTwinMuxDigiToRaw() override;

    static void fillDescriptions(edm::ConfigurationDescriptions & descs);

    void beginRun(edm::Run const & run, edm::EventSetup const & setup) override;
    void produce(edm::Event & event, edm::EventSetup const & setup) override;

protected:
    edm::EDGetTokenT<RPCDigiCollection> digi_token_;

    int bx_min_, bx_max_;
    bool ignore_eod_;
    int event_type_;
    unsigned int ufov_;

    edm::ESWatcher<RPCTwinMuxLinkMapRcd> es_tm_link_map_watcher_;
    std::map<int, std::vector<RPCAMCLink> > fed_amcs_;
};

#endif // EventFilter_RPCRawToDigi_RPCTwinMuxDigiToRaw_h
