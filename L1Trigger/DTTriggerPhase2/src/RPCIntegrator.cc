#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/DTTriggerPhase2/interface/RPCIntegrator.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <cmath>

using namespace cmsdt;

RPCIntegrator::RPCIntegrator(const edm::ParameterSet& pset, edm::ConsumesCollector& iC)
    : m_debug_(pset.getUntrackedParameter<bool>("debug")),
      m_max_quality_to_overwrite_t0_(pset.getParameter<int>("max_quality_to_overwrite_t0")),
      m_bx_window_(pset.getParameter<int>("bx_window")),
      m_phi_window_(pset.getParameter<double>("phi_window")),
      m_storeAllRPCHits_(pset.getParameter<bool>("storeAllRPCHits")) {
  if (m_debug_)
    LogDebug("RPCIntegrator") << "RPCIntegrator constructor";

  rpcGeomH_ = iC.esConsumes<RPCGeometry, MuonGeometryRecord>();
  dtGeomH_ = iC.esConsumes<DTGeometry, MuonGeometryRecord>();
}

RPCIntegrator::~RPCIntegrator() {
  if (m_debug_)
    LogDebug("RPCIntegrator") << "RPCIntegrator destructor";
}

void RPCIntegrator::initialise(const edm::EventSetup& iEventSetup, double shift_back_fromDT) {
  if (m_debug_)
    LogDebug("RPCIntegrator") << "RPCIntegrator initialisation";

  if (m_debug_)
    LogDebug("RPCIntegrator") << "Getting RPC geometry";

  if (auto handle = iEventSetup.getHandle(dtGeomH_)) {
    dtGeo_ = handle.product();
  }

  if (auto handle = iEventSetup.getHandle(rpcGeomH_)) {
    rpcGeo_ = handle.product();
  }

  shift_back_ = shift_back_fromDT;
}

void RPCIntegrator::finish() {}

void RPCIntegrator::prepareMetaPrimitives(edm::Handle<RPCRecHitCollection> rpcRecHits) {
  RPCMetaprimitives_.clear();
  rpcRecHits_translated_.clear();
  for (const auto& rpcIt : *rpcRecHits) {
    RPCDetId rpcDetId = (RPCDetId)(rpcIt).rpcId();
    GlobalPoint global_position = RPCGlobalPosition(rpcDetId, rpcIt);
    int rpc_region = rpcDetId.region();
    if (rpc_region != 0)
      continue;  // Region = 0 Barrel

    // set everyone to rpc single hit (3) not matched to DT flag for now
    // change last two elements if dt bx centered at zero again
    RPCMetaprimitives_.emplace_back(
        rpcDetId, &rpcIt, global_position, RPC_HIT, rpcIt.BunchX() + BX_SHIFT, rpcIt.time() + BX_SHIFT * LHC_CLK_FREQ);
  }
}
void RPCIntegrator::matchWithDTAndUseRPCTime(std::vector<metaPrimitive>& dt_metaprimitives) {
  for (auto dt_metaprimitive = dt_metaprimitives.begin(); dt_metaprimitive != dt_metaprimitives.end();
       dt_metaprimitive++) {
    RPCMetaprimitive* bestMatch_rpcRecHit = matchDTwithRPC(&*dt_metaprimitive);
    if (bestMatch_rpcRecHit) {
      (*dt_metaprimitive).rpcFlag = RPC_CONFIRM;
      if ((*dt_metaprimitive).quality < m_max_quality_to_overwrite_t0_) {
        (*dt_metaprimitive).t0 = bestMatch_rpcRecHit->rpc_t0 + 25 * shift_back_;
        (*dt_metaprimitive).rpcFlag = RPC_TIME;
      }
    }
  }
}

void RPCIntegrator::makeRPCOnlySegments() {
  std::vector<L1Phase2MuDTPhDigi> rpc_only_segments;
  for (auto& rpc_mp_it_layer1 : RPCMetaprimitives_) {
    RPCDetId rpc_id_l1 = rpc_mp_it_layer1.rpc_id;
    const RPCRecHit* rpc_cluster_l1 = rpc_mp_it_layer1.rpc_cluster;
    GlobalPoint rpc_gp_l1 = rpc_mp_it_layer1.global_position;
    if (rpc_id_l1.station() > 2 || rpc_id_l1.layer() != 1 ||
        (rpc_mp_it_layer1.rpcFlag == RPC_ASSOCIATE && !m_storeAllRPCHits_))
      continue;
    // only one RPC layer in station three and four &&
    // avoid duplicating pairs &&
    // avoid building RPC only segment if DT segment was already there
    int min_dPhi = std::numeric_limits<int>::max();
    RPCMetaprimitive* bestMatch_rpc_mp_layer2 = nullptr;
    for (auto& rpc_mp_it_layer2 : RPCMetaprimitives_) {
      RPCDetId rpc_id_l2 = rpc_mp_it_layer2.rpc_id;
      const RPCRecHit* rpc_cluster_l2 = rpc_mp_it_layer2.rpc_cluster;
      GlobalPoint rpc_gp_l2 = rpc_mp_it_layer2.global_position;
      if (rpc_id_l2.station() == rpc_id_l1.station() && rpc_id_l2.ring() == rpc_id_l1.ring() &&
          rpc_id_l2.layer() != rpc_id_l1.layer()  // ensure to have layer 1 --> layer 2
          && rpc_id_l2.sector() == rpc_id_l1.sector() && rpc_cluster_l2->BunchX() == rpc_cluster_l1->BunchX() &&
          (rpc_mp_it_layer2.rpcFlag != RPC_ASSOCIATE || m_storeAllRPCHits_)) {
        // avoid building RPC only segment with a hit already matched to DT,
        // except if one aske to store all RPC info
        float tmp_dPhi = rpc_gp_l1.phi() - rpc_gp_l2.phi();
        if (std::abs(tmp_dPhi) < std::abs(min_dPhi)) {
          min_dPhi = tmp_dPhi;
          bestMatch_rpc_mp_layer2 = &rpc_mp_it_layer2;
        }
      }
    }
    if (bestMatch_rpc_mp_layer2) {
      rpc_mp_it_layer1.rpcFlag = 6;
      // need a new flag (will be removed later) to differentiate
      // between "has been matched to DT" and "Has been used in an RPC only segment"
      bestMatch_rpc_mp_layer2->rpcFlag = 6;
      double phiB = phiBending(&rpc_mp_it_layer1, &*bestMatch_rpc_mp_layer2);
      // Arbitrarily choose the phi from layer 1
      double global_phi = rpc_mp_it_layer1.global_position.phi();
      double t0 = (rpc_mp_it_layer1.rpc_t0 + bestMatch_rpc_mp_layer2->rpc_t0) / 2;
      // RPC only segment have rpcFlag==2
      L1Phase2MuDTPhDigi rpc_only_segment =
          createL1Phase2MuDTPhDigi(rpc_id_l1, rpc_mp_it_layer1.rpc_bx, t0, global_phi, phiB, 2);
      rpc_only_segments.push_back(rpc_only_segment);
    }
  }
  rpcRecHits_translated_.insert(rpcRecHits_translated_.end(), rpc_only_segments.begin(), rpc_only_segments.end());
}

void RPCIntegrator::storeRPCSingleHits() {
  for (auto rpc_mp_it = RPCMetaprimitives_.begin(); rpc_mp_it != RPCMetaprimitives_.end(); rpc_mp_it++) {
    RPCDetId rpcDetId = rpc_mp_it->rpc_id;
    if (rpc_mp_it->rpcFlag == 6)
      rpc_mp_it->rpcFlag = RPC_ASSOCIATE;
    L1Phase2MuDTPhDigi rpc_out = createL1Phase2MuDTPhDigi(
        rpcDetId, rpc_mp_it->rpc_bx, rpc_mp_it->rpc_t0, rpc_mp_it->global_position.phi(), -10000, rpc_mp_it->rpcFlag);
    rpcRecHits_translated_.push_back(rpc_out);
  }
}

void RPCIntegrator::removeRPCHitsUsed() {
  if (m_debug_)
    LogDebug("RPCIntegrator") << "RPCIntegrator removeRPCHitsUsed method";
  if (!m_storeAllRPCHits_) {
    // Remove RPC hit attached to a DT or RPC segment if required by user
    // (avoid having two TP's corresponding to the same physical hit)
    auto rpcRecHit_translated_ = rpcRecHits_translated_.begin();
    while (rpcRecHit_translated_ != rpcRecHits_translated_.end()) {
      if (rpcRecHit_translated_->rpcFlag() == RPC_ASSOCIATE || rpcRecHit_translated_->rpcFlag() == 6) {
        rpcRecHit_translated_ = rpcRecHits_translated_.erase(rpcRecHit_translated_);
      } else {
        ++rpcRecHit_translated_;
      }
    }
  }
}

RPCMetaprimitive* RPCIntegrator::matchDTwithRPC(metaPrimitive* dt_metaprimitive) {
  // metaprimitive dtChId is still in convention with [1 - 12]
  // because at this stage the BX of metaprimitive is not yet computed...
  // will also have to subtract 20*25 ns because of the recent change
  int dt_bx = (int)round(dt_metaprimitive->t0 / 25.) - shift_back_;
  DTChamberId dt_chId = DTChamberId(dt_metaprimitive->rawId);
  int dt_sector = dt_chId.sector();
  if (dt_sector == 13)
    dt_sector = 4;
  if (dt_sector == 14)
    dt_sector = 10;
  RPCMetaprimitive* bestMatch_rpcRecHit = nullptr;
  float min_dPhi = std::numeric_limits<float>::max();
  for (auto rpc_mp_it = RPCMetaprimitives_.begin(); rpc_mp_it != RPCMetaprimitives_.end(); rpc_mp_it++) {
    RPCDetId rpc_det_id = rpc_mp_it->rpc_id;
    if (rpc_det_id.ring() == dt_chId.wheel()  // ring() in barrel RPC corresponds to the wheel
        && rpc_det_id.station() == dt_chId.station() && rpc_det_id.sector() == dt_sector &&
        std::abs(rpc_mp_it->rpc_bx - dt_bx) <= m_bx_window_) {
      // Select the RPC hit closest in phi to the DT meta primitive

      // just a trick to apply the phi window cut on what could be accessed to fine tune it
      int delta_phi =
          (int)round((phi_DT_MP_conv(rpc_mp_it->global_position.phi(), rpc_det_id.sector()) - dt_metaprimitive->phi) *
                     cmsdt::PHIBRES_CONV);
      if (std::abs(delta_phi) < min_dPhi && std::abs(delta_phi) < m_phi_window_) {
        min_dPhi = std::abs(delta_phi);
        bestMatch_rpcRecHit = &*rpc_mp_it;
      }
    }
  }
  if (bestMatch_rpcRecHit) {
    bestMatch_rpcRecHit->rpcFlag = RPC_ASSOCIATE;
  }
  return bestMatch_rpcRecHit;
}

L1Phase2MuDTPhDigi RPCIntegrator::createL1Phase2MuDTPhDigi(
    RPCDetId rpcDetId, int rpc_bx, double rpc_time, double rpc_global_phi, double phiB, int rpc_flag) {
  if (m_debug_)
    LogDebug("RPCIntegrator") << "Creating DT TP out of RPC recHits";
  int rpc_wheel = rpcDetId.ring();             // In barrel, wheel is accessed via ring() method ([-2,+2])
  int trigger_sector = rpcDetId.sector() - 1;  // DT Trigger sector:[0,11] while RPC sector:[1,12]
  int rpc_station = rpcDetId.station();
  int rpc_layer = rpcDetId.layer();
  int rpc_trigger_phi = phiInDTTPFormat(rpc_global_phi, rpcDetId.sector());
  int rpc_trigger_phiB = (phiB == -10000) ? phiB : (int)round(phiB * cmsdt::PHIBRES_CONV);
  int rpc_quality = -1;  // dummy for rpc
  int rpc_index = 0;     // dummy for rpc
  return L1Phase2MuDTPhDigi(rpc_bx,
                            rpc_wheel,
                            trigger_sector,
                            rpc_station,
                            rpc_layer,  //this would be the layer in the new dataformat
                            rpc_trigger_phi,
                            rpc_trigger_phiB,
                            rpc_quality,
                            rpc_index,
                            rpc_time,
                            -1,  // no chi2 for RPC
                            rpc_flag);
}

double RPCIntegrator::phiBending(RPCMetaprimitive* rpc_hit_1, RPCMetaprimitive* rpc_hit_2) {
  DTChamberId DT_chamber(rpc_hit_1->rpc_id.ring(), rpc_hit_1->rpc_id.station(), rpc_hit_1->rpc_id.sector());
  LocalPoint lp_rpc_hit_1_dtconv = dtGeo_->chamber(DT_chamber)->toLocal(rpc_hit_1->global_position);
  LocalPoint lp_rpc_hit_2_dtconv = dtGeo_->chamber(DT_chamber)->toLocal(rpc_hit_2->global_position);
  double slope = (lp_rpc_hit_1_dtconv.x() - lp_rpc_hit_2_dtconv.x()) / distance_between_two_rpc_layers_;
  double average_x = (lp_rpc_hit_1_dtconv.x() + lp_rpc_hit_2_dtconv.x()) / 2;
  GlobalPoint seg_middle_global =
      dtGeo_->chamber(DT_chamber)->toGlobal(LocalPoint(average_x, 0., 0.));  // for station 1 and 2, z = 0
  double seg_phi = phi_DT_MP_conv(seg_middle_global.phi(), rpc_hit_1->rpc_id.sector());
  double psi = atan(slope);
  double phiB = hasPosRF_rpc(rpc_hit_1->rpc_id.ring(), rpc_hit_1->rpc_id.sector()) ? psi - seg_phi : -psi - seg_phi;
  return phiB;
}

int RPCIntegrator::phiInDTTPFormat(double rpc_global_phi, int rpcSector) {
  double rpc_localDT_phi;
  rpc_localDT_phi = phi_DT_MP_conv(rpc_global_phi, rpcSector) * cmsdt::PHIBRES_CONV;
  return (int)round(rpc_localDT_phi);
}

double RPCIntegrator::phi_DT_MP_conv(double rpc_global_phi, int rpcSector) {
  // Adaptation of https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTwinMux/src/RPCtoDTTranslator.cc#L349

  if (rpcSector == 1)
    return rpc_global_phi;
  else {
    float conversion = 1 / 6.;
    if (rpc_global_phi >= 0)
      return rpc_global_phi - (rpcSector - 1) * M_PI * conversion;
    else
      return rpc_global_phi + (13 - rpcSector) * M_PI * conversion;
  }
}

GlobalPoint RPCIntegrator::RPCGlobalPosition(RPCDetId rpcId, const RPCRecHit& rpcIt) const {
  RPCDetId rpcid = RPCDetId(rpcId);
  const LocalPoint& rpc_lp = rpcIt.localPosition();
  const GlobalPoint& rpc_gp = rpcGeo_->idToDet(rpcid)->surface().toGlobal(rpc_lp);

  return rpc_gp;
}

bool RPCIntegrator::hasPosRF_rpc(int wh, int sec) const { return (wh > 0 || (wh == 0 && sec % 4 > 1)); }
