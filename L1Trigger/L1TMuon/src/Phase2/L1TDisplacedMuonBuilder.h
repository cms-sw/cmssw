#ifndef L1Trigger_L1TMuon_L1TDisplacedMuonBuilder
#define L1Trigger_L1TMuon_L1TDisplacedMuonBuilder

#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// helper classes
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitiveFwd.h"
#include "L1Trigger/L1TMuon/src/Phase2/L1TDisplacedMuonPtAssignment.h"
#include "L1Trigger/L1TMuon/src/Phase2/L1TDisplacedMuonStubRecovery.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCComparatorDigiFitter.h"

// stubs
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"

// tracks
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;
class DTGeometry;
class ME0Geometry;

namespace L1TMuon {

  class L1TDisplacedMuonBuilder {
  public:

    L1TDisplacedMuonBuilder(const edm::ParameterSet& iConfig);
    ~L1TDisplacedMuonBuilder();

    // trigge geometry
    void setCSCGeometry(const CSCGeometry *g) { cscGeometry_ = g; }
    void setGEMGeometry(const GEMGeometry *g) { gemGeometry_ = g; }
    void setRPCGeometry(const RPCGeometry *g) { rpcGeometry_ = g; }
    void setDTGeometry(const DTGeometry *g) { dtGeometry_ = g; }
    void setME0Geometry(const ME0Geometry *g) { me0Geometry_ = g; }

    // main function
    void build(const CSCComparatorDigiCollection*,
               const CSCCorrelatedLCTDigiCollection*,
               const GEMPadDigiCollection*,
               const GEMCoPadDigiCollection*,
               const ME0SegmentCollection*,
               const l1t::EMTFTrackCollection*,
               const L1MuBMTrackCollection*,
               const edm::Handle<l1t::MuonBxCollection>&,
               std::unique_ptr<l1t::MuonBxCollection>&);

    void fitComparatorDigis(bool fit) {fitComparatorDigis_ = fit;}
    void fitStubs(bool fit) {fitStubs_ = fit;}
    void useGE21(bool use) {useGE21_ = use;}
    void useME0(bool use) {useME0_ = use;}

  private:

    bool doRPCStubRecovery_;
    bool doCSCStubRecovery_;
    bool doDTStubRecovery_;
    bool doGEMStubRecovery_;
    bool doME0StubRecovery_;

    bool fitComparatorDigis_;
    bool fitStubs_;
    bool doStubRecovery_;
    bool useGE21_;
    bool useME0_;

    // trigger geometry
    const CSCGeometry* cscGeometry_;
    const GEMGeometry* gemGeometry_;
    const RPCGeometry* rpcGeometry_;
    const DTGeometry* dtGeometry_;
    const ME0Geometry* me0Geometry_;

    /* std::vector<L1TMuon::TriggerPrimitive> stubs_; */
    std::vector<GEMPadDigiId> pads_;
    std::vector<CSCCorrelatedLCTDigiId> lcts_;
    std::vector<L1MuDTChambPhDigiId> dts_;

    // helper classes
    std::unique_ptr<CSCComparatorDigiFitter> fitter_;
    std::unique_ptr<L1TDisplacedMuonStubRecovery> recovery_;
    /* std::unique_ptr<L1TDisplacedMuonDirectionCalculator> calculator_; */
    std::unique_ptr<L1TMuon::L1TDisplacedMuonPtAssignment> assignment_;

    // L1Mu properties
    int charge;
    int nstubs;
    bool hasCSC_[4] = {false, false, false, false};
    bool hasRPC_[4] = {false, false, false, false};
    bool hasDT_[4] = {false, false, false, false};
    bool hasGEM_[4] = {false, false, false, false};
    bool hasME0_ = false;

    // stub properties
    std::map<int, std::vector<float>> cscFitPhiLayers_;
    std::map<int, std::vector<float>> cscFitZLayers_;
    std::map<int, float> cscFitRLayers_;

    enum MuonType{Barrel, Overlap, EndcapLow, EndcapMedium, EndcapHigh};
  };
}

#endif
