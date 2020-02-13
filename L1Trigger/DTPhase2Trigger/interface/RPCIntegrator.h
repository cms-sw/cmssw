#ifndef Phase2L1Trigger_DTTrigger_RPCIntegrator_cc
#define Phase2L1Trigger_DTTrigger_RPCIntegrator_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

//DT geometry
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"

struct rpc_metaprimitive {
    RPCDetId rpc_id;
    const RPCRecHit* rpc_cluster;
    GlobalPoint global_position;
    int rpcFlag;
    int rpc_bx;
    double rpc_t0;
    rpc_metaprimitive(RPCDetId rpc_id_construct, const RPCRecHit* rpc_cluster_construct, GlobalPoint global_position_construct, int rpcFlag_construct, int rpc_bx_construct, double rpc_t0_construct) : rpc_id(rpc_id_construct), rpc_cluster(rpc_cluster_construct), global_position(global_position_construct), rpcFlag(rpcFlag_construct), rpc_bx(rpc_bx_construct), rpc_t0(rpc_t0_construct) { }
};

// change to this one if DT bx centered at zero again
//struct rpc_metaprimitive {
//    RPCDetId rpc_id;
//    const RPCRecHit* rpc_cluster;
//    GlobalPoint global_position;
//    int rpcFlag;
//    rpc_metaprimitive(RPCDetId rpc_id_construct, const RPCRecHit* rpc_cluster_construct, GlobalPoint global_position_construct, int rpcFlag_construct) : rpc_id(rpc_id_construct), rpc_cluster(rpc_cluster_construct), global_position(global_position_construct), rpcFlag(rpcFlag_construct) { }
//};

class RPCIntegrator {
    public:
        RPCIntegrator(const edm::ParameterSet& pset);
        ~RPCIntegrator();

        void initialise(const edm::EventSetup& iEventSetup, double shift_back_fromDT);
        void finish();

        void prepareMetaPrimitives(edm::Handle<RPCRecHitCollection> rpcRecHits);
        void matchWithDTAndUseRPCTime(std::vector<metaPrimitive> & dt_metaprimitives);
        void makeRPCOnlySegments();
        void storeRPCSingleHits();
        void removeRPCHitsUsed();

        rpc_metaprimitive* matchDTwithRPC(metaPrimitive* dt_metaprimitive);
        L1Phase2MuDTPhDigi createL1Phase2MuDTPhDigi(RPCDetId rpcDetId, int rpc_bx, double rpc_time, double rpc_global_phi, double phiB, int rpc_flag);

        double getPhiBending(rpc_metaprimitive* rpc_hit_1, rpc_metaprimitive* rpc_hit_2);
        int getPhiInDTTPFormat(double rpc_global_phi, int rpcSector);
        GlobalPoint getRPCGlobalPosition(RPCDetId rpcId, const RPCRecHit& rpcIt) const;
        double getPhi_DT_MP_conv(double rpc_global_phi, int rpcSector);
        bool hasPosRF_rpc(int wh, int sec);

        std::vector<L1Phase2MuDTPhDigi> rpcRecHits_translated;
        std::vector<rpc_metaprimitive> rpc_metaprimitives;

    private:
        //RPCRecHitCollection m_rpcRecHits;
        Bool_t m_debug;
        int m_max_quality_to_overwrite_t0;
        int m_bx_window;
        double m_phi_window;
        bool m_storeAllRPCHits;
        edm::ESHandle<RPCGeometry> m_rpcGeo;
        edm::ESHandle<DTGeometry> m_dtGeo;
        double m_dt_phi_granularity = 65536. / 0.8; // 65536 different values per 0.8 radian
        double m_dt_phiB_granularity = 2048. / 1.4; // 2048. different values per 1.4 radian
        // Constant geometry values
        //R[stat][layer] - radius of rpc station/layer from center of CMS
        double R[2][2] = {{410.0, 444.8}, {492.7, 527.3}};
        double distance_between_two_rpc_layers = 35; // in cm
        double shift_back;
        //float m_radius_rb1_layer1 = 410.0, m_radius_rb1_layer2 = 444.8, m_radius_rb2_layer1 = 492.7, m_radius_rb2_layer2 = 527.3;
};
#endif
