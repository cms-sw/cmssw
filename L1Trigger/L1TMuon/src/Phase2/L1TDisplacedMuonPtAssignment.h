#ifndef L1Trigger_L1TMuon_L1TDisplacedMuonPtAssignment
#define L1Trigger_L1TMuon_L1TDisplacedMuonPtAssignment

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
#include "L1Trigger/L1TMuon/src/Phase2/GeometryHelpers.h"
#include "L1Trigger/L1TMuon/src/Phase2/EndcapTriggerPtAssignmentHelper.h"
#include "L1Trigger/L1TMuon/src/Phase2/BarrelTriggerPtAssignmentHelper.h"

class CSCGeometry;
class GEMGeometry;
class RPCGeometry;
class DTGeometry;
class ME0Geometry;

namespace L1TMuon
{
  class L1TDisplacedMuonPtAssignment
  {
  public:

    enum PtAssignmentMethod{Position, Direction, Hybrid};
    enum EndcapEtaSection{Eta12To14, Eta14To16, Eta16To18,
                          Eta18To20, Eta20To22, Eta22To24};

    L1TDisplacedMuonPtAssignment(const edm::ParameterSet& iConfig);
    ~L1TDisplacedMuonPtAssignment();

    // trigger geometry
    void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
    void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }
    void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }
    void setDTGeometry(const DTGeometry *g) { dt_g = g; }
    void setME0Geometry(const ME0Geometry *g) { me0_g = g; }

    // set inputs: muon and  trigger primitives
    void setMuon(const l1t::Muon& muon) {muon_ = muon;}
    void setTriggerPrimitives(const std::vector<GEMPadDigiId>& pads) {pads_ = pads;}
    void setTriggerPrimitives(const std::vector<CSCCorrelatedLCTDigiId>& lcts) {lcts_ = lcts;}
    void setTriggerPrimitives(const ME0Segment& seg) {segment_ = seg;}

    // specify method
    void useGE11(bool useGE11) {useGE11_ = useGE11;}
    void useGE21(bool useGE21) {useGE21_ = useGE21;}
    void useME0(bool useME0) {useME0_ = useME0;}
    void setPtAssignmentMethod(int method = 0) {method_ = method;}

    /*
      Low: 1.2 to 1.5
      Medium: 1.5 to 2.1
      High: 2.1 to 2.4
     */
    void calculatePositionPtBarrel();
    void calculatePositionPtOverlap();
    void calculatePositionPtEndcap();

    void calculateDirectionPtBarrel();
    void calculateDirectionPtOverlap();
    void calculateDirectionPtEndcapLow();
    void calculateDirectionPtEndcapMedium();
    void calculateDirectionPtEndcapHigh();

    void calculateHybridPtBarrel();
    void calculateHybridPtOverlap();
    void calculateHybridPtEndcapLow();
    void calculateHybridPtEndcapMedium();
    void calculateHybridPtEndcapHigh();

    int getBarrelStubCase(bool MB1, bool MB2, bool MB3, bool MB4);

  private:

    bool useGE11_;
    bool useGE21_;
    bool useME0_;
    int method_;

    // trigger geometry
    const CSCGeometry* csc_g;
    const GEMGeometry* gem_g;
    const RPCGeometry* rpc_g;
    const DTGeometry* dt_g;
    const ME0Geometry* me0_g;

    l1t::Muon muon_;
    std::vector<GEMPadDigiId> pads_;
    std::vector<CSCCorrelatedLCTDigiId> lcts_;
    ME0Segment segment_;

    // Barrel members
    bool has_stub_mb1, has_stub_mb2, has_stub_mb3, has_stub_mb4;
    float phi_mb1, phi_mb2, phi_mb3, phi_mb4;
    float phib_mb1, phib_mb2, phib_mb3, phib_mb4;
    float dphi_mb1, dphi_mb2, dphi_mb3, dphi_mb4; //phi + phib
    float dPhi_barrel_dir_12, dPhi_barrel_dir_13, dPhi_barrel_dir_14;
    float dPhi_barrel_dir_23, dPhi_barrel_dir_24, dPhi_barrel_dir_34;

    // Endcap members
    bool hasCSC_[4] = {false, false, false, false};
    bool hasRPC_[4] = {false, false, false, false};
    bool hasDT_[4] = {false, false, false, false};
    bool hasGEM_[4] = {false, false, false, false};

    EndcapTriggerPtAssignmentHelper::EvenOdd isEven[4] = {EndcapTriggerPtAssignmentHelper::EvenOdd::Odd,
                                                          EndcapTriggerPtAssignmentHelper::EvenOdd::Odd,
                                                          EndcapTriggerPtAssignmentHelper::EvenOdd::Odd,
                                                          EndcapTriggerPtAssignmentHelper::EvenOdd::Odd};
    GlobalPoint gp_ME[4];
    GlobalPoint gp_st_layer3[4];

    // pts
    float positionPt_;
    float directionPt_;
    float hybridPt_;
  };
}

#endif
