#ifndef L1Trigger_L1TMuon_L1TDisplacedMuonStubRecovery
#define L1Trigger_L1TMuon_L1TDisplacedMuonStubRecovery

#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "L1Trigger/L1TMuon/src/Phase2/GeometryHelpers.h"

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;
class DTGeometry;
class ME0Geometry;

namespace L1TMuon {

  class L1TDisplacedMuonStubRecovery {
  public:
    L1TDisplacedMuonStubRecovery(const edm::ParameterSet& iConfig);
    ~L1TDisplacedMuonStubRecovery();

    // trigger geometry
    void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
    void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }
    void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }
    void setDTGeometry(const DTGeometry *g)   { dt_g = g; }
    void setME0Geometry(const ME0Geometry *g) { me0_g = g; }

    // recover missing CSC stubs
    void recoverCSCLCT(const l1t::EMTFTrack&,
                       const l1t::EMTFTrackCollection* emtfTracks,
                       const CSCCorrelatedLCTDigiCollection* lcts,
                       int station,
                       CSCCorrelatedLCTDigiId& bestLCT) const;

    // is a stub already part of a track?
    bool stubInEMTFTracks(const CSCCorrelatedLCTDigi& stub,
                          const l1t::EMTFTrackCollection& l1Tracks) const;

    // match L1Mu to ME0 segment
    void getBestMatchedME0(const l1t::Muon& l1mu,
                           const ME0SegmentCollection* segments,
                           ME0Segment& bestSegment) const;

  private:

    // trigger geometry
    const CSCGeometry* csc_g;
    const GEMGeometry* gem_g;
    const RPCGeometry* rpc_g;
    const DTGeometry* dt_g;
    const ME0Geometry* me0_g;

    int verbose_;
    float max_dR_L1Mu_ME0_;
  };
}

#endif
