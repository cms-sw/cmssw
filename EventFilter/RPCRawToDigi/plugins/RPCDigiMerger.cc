#include "EventFilter/RPCRawToDigi/plugins/RPCDigiMerger.h"

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

using namespace edm;
using namespace std;

RPCDigiMerger::RPCDigiMerger(edm::ParameterSet const & config)
{
    produces<RPCDigiCollection>();
    // if (fill_counters_) {
    //     produces<RPCAMCLinkCounters>();
    // }
    TwinMux_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagTwinMuxDigis"));
    OMTF_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagOMTFDigis"));
    CPPF_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagCPPFDigis"));
}

RPCDigiMerger::~RPCDigiMerger()
{}

void RPCDigiMerger::fillDescriptions(edm::ConfigurationDescriptions & descs)
{
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputTagTwinMuxDigis", edm::InputTag("RPCTwinMuxRawToDigi", ""));
    desc.add<edm::InputTag>("inputTagOMTFDigis", edm::InputTag("omtfStage2Digis", ""));
    desc.add<edm::InputTag>("inputTagCPPFDigis", edm::InputTag("rpcCPPFRawToDigi", ""));

    descs.add("RPCDigiMerger", desc);
}

void RPCDigiMerger::beginRun(edm::Run const & run, edm::EventSetup const & setup)
{}

void RPCDigiMerger::produce(edm::Event & event, edm::EventSetup const & setup)
{

    // Get the digis 
    // TwinMux
    Handle<RPCDigiCollection> TwinMux_digis; 
    event.getByToken(TwinMux_token_,TwinMux_digis);
    // OMTF
    Handle<RPCDigiCollection> OMTF_digis; 
    event.getByToken(OMTF_token_,OMTF_digis);
    // CPFF
    Handle<RPCDigiCollection> CPPF_digis; 
    event.getByToken(CPPF_token_,CPPF_digis);

    // new RPCDigiCollection
    std::unique_ptr<RPCDigiCollection> rpc_digi_collection(new RPCDigiCollection());

    // loop over twinmux digis
    for ( auto rpcdgIt = TwinMux_digis->begin(); rpcdgIt != TwinMux_digis->end(); ++rpcdgIt ) {
        // The layerId
        const RPCDetId& rpcId = (*rpcdgIt).first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = (*rpcdgIt).second;

        rpc_digi_collection->put(range, rpcId);
    }

    // loop over cpppf digis
    for ( auto rpcdgIt = CPPF_digis->begin(); rpcdgIt != CPPF_digis->end(); ++rpcdgIt ) {
        // The layerId
        const RPCDetId& rpcId = (*rpcdgIt).first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = (*rpcdgIt).second;

        rpc_digi_collection->put(range, rpcId);
    }

    // loop over omtf digis
    for ( auto rpcdgIt = OMTF_digis->begin(); rpcdgIt != OMTF_digis->end(); ++rpcdgIt ) {
        // The layerId
        const RPCDetId& rpcId = (*rpcdgIt).first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = (*rpcdgIt).second;

        if ( ((rpcId.region() == -1 || rpcId.region() == 1) && (rpcId.ring() == 3) && (rpcId.station() == 1 || rpcId.station() == 2)) )  {
            // accepts only rings: RE-2_R3 ; RE-1_R3 ; RE+1_R3 ; RE+2_R3 ; 
            rpc_digi_collection->put(range, rpcId);
        }
    }

    // "put" into the event
    event.put(std::move(rpc_digi_collection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCDigiMerger);
