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

RPCDigiMerger::RPCDigiMerger(edm::ParameterSet const& config)
    : bx_minTwinMux_(config.getParameter<int>("bxMinTwinMux")),
      bx_maxTwinMux_(config.getParameter<int>("bxMaxTwinMux")),
      bx_minOMTF_(config.getParameter<int>("bxMinOMTF")),
      bx_maxOMTF_(config.getParameter<int>("bxMaxOMTF")),
      bx_minCPPF_(config.getParameter<int>("bxMinCPPF")),
      bx_maxCPPF_(config.getParameter<int>("bxMaxCPPF")) {
  produces<RPCDigiCollection>();
  simRPC_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagSimRPCDigis"));
  // protection against empty InputTag to allow for Data/MC compatibility
  if (not config.getParameter<edm::InputTag>("inputTagTwinMuxDigis").label().empty()) {
    twinMux_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagTwinMuxDigis"));
  }
  if (not config.getParameter<edm::InputTag>("inputTagOMTFDigis").label().empty()) {
    omtf_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagOMTFDigis"));
  }
  if (not config.getParameter<edm::InputTag>("inputTagCPPFDigis").label().empty()) {
    cppf_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTagCPPFDigis"));
  }
}

RPCDigiMerger::~RPCDigiMerger() {}

void RPCDigiMerger::fillDescriptions(edm::ConfigurationDescriptions& descs) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTagSimRPCDigis", edm::InputTag("simMuonRPCDigis", ""));
  desc.add<edm::InputTag>("inputTagTwinMuxDigis", edm::InputTag("", ""));
  desc.add<edm::InputTag>("inputTagOMTFDigis", edm::InputTag("", ""));
  desc.add<edm::InputTag>("inputTagCPPFDigis", edm::InputTag("", ""));
  desc.add<edm::InputTag>("InputLabel", edm::InputTag(" "));
  desc.add<int>("bxMinTwinMux", -2);
  desc.add<int>("bxMaxTwinMux", 2);
  desc.add<int>("bxMinOMTF", -3);
  desc.add<int>("bxMaxOMTF", 4);
  desc.add<int>("bxMinCPPF", -2);
  desc.add<int>("bxMaxCPPF", 2);

  descs.add("rpcDigiMerger", desc);
}

void RPCDigiMerger::beginRun(edm::Run const& run, edm::EventSetup const& setup) {}

void RPCDigiMerger::produce(edm::Event& event, edm::EventSetup const& setup) {
  // Get the digis
  // new RPCDigiCollection
  std::unique_ptr<RPCDigiCollection> rpc_digi_collection(new RPCDigiCollection());

  //Check if its Data
  if (not(cppf_token_.isUninitialized() && omtf_token_.isUninitialized() && twinMux_token_.isUninitialized())) {
    // loop over TwinMux digis
    // protection against empty InputTag to allow for Data/MC compatibility
    if (not twinMux_token_.isUninitialized()) {
      Handle<RPCDigiCollection> TwinMux_digis;
      event.getByToken(twinMux_token_, TwinMux_digis);
      for (const auto&& rpcdgIt : (*TwinMux_digis)) {
        // The layerId
        const RPCDetId& rpcId = rpcdgIt.first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = rpcdgIt.second;
        rpc_digi_collection->put(range, rpcId);
      }
    }
    // loop over CPPF digis
    // protection against empty InputTag to allow for Data/MC compatibility
    if (not cppf_token_.isUninitialized()) {
      Handle<RPCDigiCollection> CPPF_digis;
      event.getByToken(cppf_token_, CPPF_digis);
      for (const auto&& rpcdgIt : (*CPPF_digis)) {
        // The layerId
        const RPCDetId& rpcId = rpcdgIt.first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = rpcdgIt.second;
        rpc_digi_collection->put(range, rpcId);
      }
    }
    // loop over OMTF digis
    // protection against empty InputTag to allow for Data/MC compatibility
    if (not omtf_token_.isUninitialized()) {
      Handle<RPCDigiCollection> OMTF_digis;
      event.getByToken(omtf_token_, OMTF_digis);
      for (const auto& rpcdgIt : (*OMTF_digis)) {
        // The layerId
        const RPCDetId& rpcId = rpcdgIt.first;
        // Get the iterators over the digis associated with this LayerId
        const RPCDigiCollection::Range& range = rpcdgIt.second;
        // accepts only rings: RE-2_R3 ; RE-1_R3 ; RE+1_R3 ; RE+2_R3 ;
        if (((rpcId.region() == -1 || rpcId.region() == 1) && (rpcId.ring() == 3) &&
             (rpcId.station() == 1 || rpcId.station() == 2))) {
          rpc_digi_collection->put(range, rpcId);
        }
      }
    }
  } else {  //its MC
    // SimRPCDigis collection
    Handle<RPCDigiCollection> SimRPC_digis;
    event.getByToken(simRPC_token_, SimRPC_digis);

    RPCDetId rpc_det_id;
    std::vector<RPCDigi> local_rpc_digis;

    // loop over SimRPC digis
    for (const auto& rpc_digi : (*SimRPC_digis)) {
      // The layerId
      const RPCDetId& rpcId = rpc_digi.first;
      // Get the iterators over the digis associated with this LayerId
      const RPCDigiCollection::Range& range = rpc_digi.second;

      if (rpcId != rpc_det_id) {
        if (!local_rpc_digis.empty()) {
          rpc_digi_collection->put(RPCDigiCollection::Range(local_rpc_digis.begin(), local_rpc_digis.end()),
                                   rpc_det_id);
          local_rpc_digis.clear();
        }
        rpc_det_id = rpcId;
      }
      for (std::vector<RPCDigi>::const_iterator id = range.first; id != range.second; id++) {
        const RPCDigi& dit = (*id);
        //Barrel
        if (rpcId.region() == 0) {
          //TwinMux
          if (dit.bx() >= bx_minTwinMux_ && dit.bx() <= bx_maxTwinMux_) {
            local_rpc_digis.push_back(dit);
          }
        }
        //EndCap
        if (rpcId.region() == -1 || rpcId.region() == 1) {
          //OMTF
          if (rpcId.ring() == 3 && (rpcId.station() == 1 || rpcId.station() == 2) && dit.bx() >= bx_minOMTF_ &&
              dit.bx() <= bx_maxOMTF_) {
            local_rpc_digis.push_back(dit);
          }
          //CPPF
          if (((rpcId.ring() == 2) || (rpcId.ring() == 3 && (rpcId.station() == 3 || rpcId.station() == 4))) &&
              (dit.bx() >= bx_minCPPF_ && dit.bx() <= bx_maxCPPF_)) {
            local_rpc_digis.push_back(dit);
          }
        }
      }
    }
    if (!local_rpc_digis.empty()) {
      rpc_digi_collection->put(RPCDigiCollection::Range(local_rpc_digis.begin(), local_rpc_digis.end()), rpc_det_id);
    }
  }
  // "put" into the event
  event.put(std::move(rpc_digi_collection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCDigiMerger);
