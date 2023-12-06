#ifndef L1Trigger_L1TMuonEndCapPhase2_TPrimitives_h
#define L1Trigger_L1TMuonEndCapPhase2_TPrimitives_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/RPCUtils.h"

namespace emtf::phase2 {

    enum TPSelection {kNative, kNeighbor, kNone};

    struct TPInfo {
        // Id
        int hit_id = -1;
        int segment_id = -1;

        // Selection
        int bx = -999;
        int ilink = -1;
        TPSelection selection = kNone;

        // Flags
        bool flag_substitute = false;

        // Detector
        int endcap = 0;
        int endcap_pm = 0;
        int sector = 0;
        int subsector = 0;
        int station = 0;
        int ring = 0;
        int roll = 0;
        int layer = 0;
        int chamber = 0;

        // CSC
        int csc_id = -1;
        csc::Facing csc_facing = csc::Facing::kNone;
        int csc_first_wire = -1;
        int csc_second_wire = -1;

        // RPC
        rpc::Type rpc_type = rpc::kNone;
    };

    class TPEntry {

        public:
            TPEntry(const TPEntry&);
            TPEntry(const TriggerPrimitive&);
            TPEntry(const TriggerPrimitive&, const TPInfo&);
            TPEntry(const CSCDetId&, const CSCCorrelatedLCTDigi&);
            TPEntry(const RPCDetId&, const RPCRecHit&);
            TPEntry(const GEMDetId&, const GEMPadDigiCluster&);
            TPEntry(const ME0DetId&, const ME0TriggerDigi&);
            TPEntry(const GEMDetId&, const ME0TriggerDigi&);

            ~TPEntry();

            TriggerPrimitive tp_;
            TPInfo info_;
    };

}

#endif  // L1Trigger_L1TMuonEndCapPhase2_TPrimitives_h not defined
