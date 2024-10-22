#ifndef Phase2L1Trigger_DTTrigger_RPCIntegrator_h
#define Phase2L1Trigger_DTTrigger_RPCIntegrator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

//DT geometry
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

struct RPCMetaprimitive {
  RPCDetId rpc_id;
  const RPCRecHit* rpc_cluster;
  GlobalPoint global_position;
  int rpcFlag;
  int rpc_bx;
  double rpc_t0;
  RPCMetaprimitive(RPCDetId rpc_id_construct,
                   const RPCRecHit* rpc_cluster_construct,
                   GlobalPoint global_position_construct,
                   int rpcFlag_construct,
                   int rpc_bx_construct,
                   double rpc_t0_construct)
      : rpc_id(rpc_id_construct),
        rpc_cluster(rpc_cluster_construct),
        global_position(global_position_construct),
        rpcFlag(rpcFlag_construct),
        rpc_bx(rpc_bx_construct),
        rpc_t0(rpc_t0_construct) {}
};

class RPCIntegrator {
public:
  RPCIntegrator(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  ~RPCIntegrator();

  void initialise(const edm::EventSetup& iEventSetup, double shift_back_fromDT);
  void finish();

  void prepareMetaPrimitives(edm::Handle<RPCRecHitCollection> rpcRecHits);
  void matchWithDTAndUseRPCTime(std::vector<cmsdt::metaPrimitive>& dt_metaprimitives);
  void makeRPCOnlySegments();
  void storeRPCSingleHits();
  void removeRPCHitsUsed();

  RPCMetaprimitive* matchDTwithRPC(cmsdt::metaPrimitive* dt_metaprimitive);
  L1Phase2MuDTPhDigi createL1Phase2MuDTPhDigi(
      RPCDetId rpcDetId, int rpc_bx, double rpc_time, double rpc_global_phi, double phiB, int rpc_flag);

  double phiBending(RPCMetaprimitive* rpc_hit_1, RPCMetaprimitive* rpc_hit_2);
  int phiInDTTPFormat(double rpc_global_phi, int rpcSector);
  GlobalPoint RPCGlobalPosition(RPCDetId rpcId, const RPCRecHit& rpcIt) const;
  double phi_DT_MP_conv(double rpc_global_phi, int rpcSector);
  bool hasPosRF_rpc(int wh, int sec) const;

  std::vector<L1Phase2MuDTPhDigi> rpcRecHits_translated_;
  std::vector<RPCMetaprimitive> RPCMetaprimitives_;

private:
  const bool m_debug_;
  int m_max_quality_to_overwrite_t0_;
  int m_bx_window_;
  double m_phi_window_;
  bool m_storeAllRPCHits_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomH_;

  DTGeometry const* dtGeo_;
  RPCGeometry const* rpcGeo_;

  // Constant geometry values
  //R[stat][layer] - radius of rpc station/layer from center of CMS
  static constexpr double R_[2][2] = {{410.0, 444.8}, {492.7, 527.3}};
  static constexpr double distance_between_two_rpc_layers_ = 35;  // in cm

  double shift_back_;
};
#endif
