#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "L1Trigger/DTPhase2Trigger/interface/RPCIntegrator.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <math.h>

RPCIntegrator::RPCIntegrator(const edm::ParameterSet& pset){
    m_debug = pset.getUntrackedParameter<Bool_t>("debug");
    if (m_debug) std::cout <<"RPCIntegrator constructor" << std::endl;
    m_max_quality_to_overwrite_t0 = pset.getUntrackedParameter<int>("max_quality_to_overwrite_t0");
    m_bx_window = pset.getUntrackedParameter<int>("bx_window");
    m_phi_window = pset.getUntrackedParameter<double>("phi_window");
    m_storeAllRPCHits = pset.getUntrackedParameter<bool>("storeAllRPCHits");
}

RPCIntegrator::~RPCIntegrator() {
    if (m_debug) std::cout <<"RPCIntegrator destructor" << std::endl;
}

void RPCIntegrator::initialise(const edm::EventSetup& iEventSetup, double shift_back_fromDT) {
    if (m_debug) std::cout << "RPCIntegrator initialisation" << std::endl;
    
    if (m_debug) std::cout << "Getting RPC geometry" << std::endl;
    iEventSetup.get<MuonGeometryRecord>().get(m_rpcGeo);

    iEventSetup.get<MuonGeometryRecord>().get(m_dtGeo);
    shift_back = shift_back_fromDT;
}

void RPCIntegrator::finish() {
    return;
}

void RPCIntegrator::prepareMetaPrimitives(edm::Handle<RPCRecHitCollection> rpcRecHits) {
    rpc_metaprimitives.clear();
    rpcRecHits_translated.clear();
    for (auto rpcIt = rpcRecHits->begin(); rpcIt != rpcRecHits->end(); rpcIt++) {
        RPCDetId rpcDetId = (RPCDetId)(*rpcIt).rpcId();
        GlobalPoint global_position = getRPCGlobalPosition(rpcDetId, *rpcIt);
        int rpc_region = rpcDetId.region();
        if(rpc_region != 0 ) continue; // Region = 0 Barrel
        rpc_metaprimitives.push_back(rpc_metaprimitive(rpcDetId, &*rpcIt, global_position, 3, rpcIt->BunchX() + 20, rpcIt->time() + 20 * 25)); // set everyone to rpc single hit not matched to DT flag for now
        // if dt bx centered at zero again
        //rpc_metaprimitives.push_back(rpc_metaprimitive(rpcDetId, &*rpcIt, global_position, 3)); // set everyone to rpc single hit not matched to DT flag for now
    }
}
void RPCIntegrator::matchWithDTAndUseRPCTime(std::vector<metaPrimitive> & dt_metaprimitives) {
    for (auto dt_metaprimitive = dt_metaprimitives.begin(); dt_metaprimitive != dt_metaprimitives.end(); dt_metaprimitive++) {
        rpc_metaprimitive* bestMatch_rpcRecHit = matchDTwithRPC(&*dt_metaprimitive);
        if (bestMatch_rpcRecHit) {
            (*dt_metaprimitive).rpcFlag = 4;
            if ((*dt_metaprimitive).quality < m_max_quality_to_overwrite_t0){
                (*dt_metaprimitive).t0 = bestMatch_rpcRecHit->rpc_t0 + 25 * shift_back; // Overwriting t0 will propagate to BX since it is defined by round((*metaPrimitiveIt).t0/25.)-shift_back
                //(*dt_metaprimitive).t0 = bestMatch_rpcRecHit->rpc_cluster->time() + 25 * shift_back; // Overwriting t0 will propagate to BX since it is defined by round((*metaPrimitiveIt).t0/25.)-shift_back
                                                                                      // but we need to add this shift back since all RPC chamber time is centered at 0 for prompt muon
                //(*dt_metaprimitive).phiB = getPhi_DT_MP_conv(bestMatch_rpcRecHit->global_position.phi(), bestMatch_rpcRecHit->rpc_id.sector()) - dt_metaprimitive->phi; // Use to fine tune the phi matching window (just need somewhere top store the deltaPhi to plot in with the output collection
                (*dt_metaprimitive).rpcFlag = 1;
            }
        }
    }
}

void RPCIntegrator::makeRPCOnlySegments() {
    std::vector<L1Phase2MuDTPhDigi> rpc_only_segments;
    for (auto rpc_mp_it_layer1 = rpc_metaprimitives.begin(); rpc_mp_it_layer1 != rpc_metaprimitives.end(); rpc_mp_it_layer1++) {
        RPCDetId rpc_id_l1 = rpc_mp_it_layer1->rpc_id;
        const RPCRecHit* rpc_cluster_l1 = rpc_mp_it_layer1->rpc_cluster;
        GlobalPoint rpc_gp_l1 = rpc_mp_it_layer1->global_position;
        if (rpc_id_l1.station() > 2 || rpc_id_l1.layer() != 1 || (rpc_mp_it_layer1->rpcFlag == 5 && !m_storeAllRPCHits)) continue; // only one RPC layer in station three and four && avoid duplicating pairs && avoid building RPC only segment if DT segment was already there
        //if (rpc_id_l1.station() > 2 || rpc_id_l1.layer() != 1 || (rpc_mp_it_layer1->rpcFlag == 5 && !m_storeAllRPCHits) || rpc_mp_it_layer1->rpcFlag == 6) continue; // only one RPC layer in station three and four && avoid duplicating pairs && avoid building RPC only segment if DT segment was already there && allow a cluster to be used in one segment only
        int min_dPhi = std::numeric_limits<int>::max();
        rpc_metaprimitive* bestMatch_rpc_mp_layer2 = NULL;
        for (auto rpc_mp_it_layer2 = rpc_metaprimitives.begin(); rpc_mp_it_layer2 != rpc_metaprimitives.end(); rpc_mp_it_layer2++) {
            RPCDetId rpc_id_l2 = rpc_mp_it_layer2->rpc_id;
            const RPCRecHit* rpc_cluster_l2 = rpc_mp_it_layer2->rpc_cluster;
            GlobalPoint rpc_gp_l2 = rpc_mp_it_layer2->global_position;
            if (rpc_id_l2.station() == rpc_id_l1.station()
                    && rpc_id_l2.ring() == rpc_id_l1.ring()
                    && rpc_id_l2.layer() != rpc_id_l1.layer() // ensure to have layer 1 --> layer 2
                    && rpc_id_l2.sector() == rpc_id_l1.sector()
                    && rpc_cluster_l2->BunchX() == rpc_cluster_l1->BunchX()
                    && (rpc_mp_it_layer2->rpcFlag != 5 || m_storeAllRPCHits)) { // avoid building RPC only segment with a hit already matched to DT, except if one aske to store all RPC info
                    //&& rpc_mp_it_layer2->rpcFlag != 6) {
                float tmp_dPhi = rpc_gp_l1.phi() - rpc_gp_l2.phi();
                if (std::abs(tmp_dPhi) < std::abs(min_dPhi)) {
                    min_dPhi = tmp_dPhi;
                    bestMatch_rpc_mp_layer2 = &*rpc_mp_it_layer2;
                }
            }
        }
        if (bestMatch_rpc_mp_layer2) {
            rpc_mp_it_layer1->rpcFlag = 6; // need a new flag (will be removed later) to differentiate between "has been matched to DT" and "Has been used in an RPC only segment"
            bestMatch_rpc_mp_layer2->rpcFlag = 6;
            double phiB = getPhiBending(&*rpc_mp_it_layer1, &*bestMatch_rpc_mp_layer2);
            //double global_phi = (rpc_mp_it_layer1->global_position.phi() + bestMatch_rpc_mp_layer2->global_position.phi()) / 2.0; // does not work...
            // FIXME define the phi of the segment at the middle of it?
            double global_phi = rpc_mp_it_layer1->global_position.phi(); // Arbitrarily choose the phi from layer 1
            //double t0 = (rpc_mp_it_layer1->rpc_cluster->time() + bestMatch_rpc_mp_layer2->rpc_cluster->time()) / 2;
            double t0 = (rpc_mp_it_layer1->rpc_t0 + bestMatch_rpc_mp_layer2->rpc_t0) / 2;
            //L1Phase2MuDTPhDigi rpc_only_segment  = createL1Phase2MuDTPhDigi(rpc_id_l1, rpc_mp_it_layer1->rpc_cluster->BunchX(), t0, global_phi, phiB, 2); // RPC only segment have rpcFlag==2
            L1Phase2MuDTPhDigi rpc_only_segment  = createL1Phase2MuDTPhDigi(rpc_id_l1, rpc_mp_it_layer1->rpc_bx, t0, global_phi, phiB, 2); // RPC only segment have rpcFlag==2
            rpc_only_segments.push_back(rpc_only_segment);
        }
    }
    rpcRecHits_translated.insert(rpcRecHits_translated.end(), rpc_only_segments.begin(), rpc_only_segments.end());
}

void RPCIntegrator::storeRPCSingleHits() {
    for (auto rpc_mp_it = rpc_metaprimitives.begin(); rpc_mp_it != rpc_metaprimitives.end(); rpc_mp_it++){
        //const RPCRecHit* rpc_hit = rpc_mp_it->rpc_cluster;
        RPCDetId rpcDetId = rpc_mp_it->rpc_id;
        if (rpc_mp_it->rpcFlag == 6) rpc_mp_it->rpcFlag = 5;
        L1Phase2MuDTPhDigi rpc_out = createL1Phase2MuDTPhDigi(rpcDetId, rpc_mp_it->rpc_bx, rpc_mp_it->rpc_t0, rpc_mp_it->global_position.phi(), -10000, rpc_mp_it->rpcFlag);
        //L1Phase2MuDTPhDigi rpc_out = createL1Phase2MuDTPhDigi(rpcDetId, rpc_hit->BunchX(), rpc_hit->time(), rpc_mp_it->global_position.phi(), -10000, rpc_mp_it->rpcFlag);
        rpcRecHits_translated.push_back(rpc_out);
    }
}

void RPCIntegrator::removeRPCHitsUsed() {
    if (m_debug) std::cout << "RPCIntegrator removeRPCHitsUsed method" << std::endl;
    if (!m_storeAllRPCHits){ // Remove RPC hit attached to a DT or RPC segment if required by user (avoid having two TP's corresponding to the same physical hit)
        auto rpcRecHit_translated = rpcRecHits_translated.begin();
        while (rpcRecHit_translated != rpcRecHits_translated.end()) {
            if (rpcRecHit_translated->rpcFlag() == 5 || rpcRecHit_translated->rpcFlag() == 6) {
                rpcRecHit_translated = rpcRecHits_translated.erase(rpcRecHit_translated);
            }
            else {
                ++rpcRecHit_translated;
            }
        }
    }
}

rpc_metaprimitive* RPCIntegrator::matchDTwithRPC(metaPrimitive* dt_metaprimitive) {
    // metaprimitive dtChId is still in convention with [1 - 12]
    int dt_bx = (int)round(dt_metaprimitive->t0/25.) - shift_back;// because at this stage the BX of metaprimitive is not yet computed... will also have to subtract 20*25 ns because of the recent change
    DTChamberId dt_chId = DTChamberId(dt_metaprimitive->rawId);
    int dt_sector = dt_chId.sector();
    if (dt_sector == 13) dt_sector = 4;
    if (dt_sector == 14) dt_sector = 10;
    rpc_metaprimitive* bestMatch_rpcRecHit = NULL;
    float min_dPhi = std::numeric_limits<float>::max();
    for (auto rpc_mp_it = rpc_metaprimitives.begin(); rpc_mp_it != rpc_metaprimitives.end(); rpc_mp_it++) {
        RPCDetId rpc_det_id = rpc_mp_it->rpc_id;
        //const RPCRecHit* rpc_cluster_toBeMatched = rpc_mp_it->rpc_cluster;
        if (rpc_det_id.ring() == dt_chId.wheel() // ring() in barrel RPC corresponds to the wheel
                && rpc_det_id.station() == dt_chId.station()
                && rpc_det_id.sector() == dt_sector
                && std::abs(rpc_mp_it->rpc_bx - dt_bx) <= m_bx_window) {
            //FIXME improve DT/RPC matching e.g. use 2D position instead of phi, use segment direction to extrapolate, etc.
            // Select the RPC hit closest in phi to the DT meta primitive

            //int rpc_phi_dtConv = getPhi_DT_MP_conv(rpc_mp_it->global_position.phi(), rpc_det_id.sector(), false);
            int delta_phi = (int)round((getPhi_DT_MP_conv(rpc_mp_it->global_position.phi(), rpc_det_id.sector()) - dt_metaprimitive->phi) * m_dt_phiB_granularity); // just a trick to apply the phi window cut on what could be accessed to fine tune it
            //if (std::abs(rpc_phi_dtConv - dt_metaprimitive->phi) < min_dPhi && delta_phi < m_phi_window){
            if (std::abs(delta_phi) < min_dPhi && std::abs(delta_phi) < m_phi_window){
                min_dPhi = std::abs(delta_phi);
                bestMatch_rpcRecHit = &*rpc_mp_it;
            }
        }
    }
    if (bestMatch_rpcRecHit){
        bestMatch_rpcRecHit->rpcFlag = 5;
    }
    return bestMatch_rpcRecHit;
}

L1Phase2MuDTPhDigi RPCIntegrator::createL1Phase2MuDTPhDigi(RPCDetId rpcDetId, int rpc_bx, double rpc_time, double rpc_global_phi, double phiB, int rpc_flag) {
    if (m_debug) std::cout << "Creating DT TP out of RPC recHits" << std::endl;
    int rpc_wheel = rpcDetId.ring(); // In barrel, wheel is accessed via ring() method ([-2,+2])
    int trigger_sector = rpcDetId.sector()-1; // DT Trigger sector:[0,11] while RPC sector:[1,12]
    int rpc_station = rpcDetId.station();
    int rpc_layer = rpcDetId.layer();
    int rpc_trigger_phi = getPhiInDTTPFormat(rpc_global_phi, rpcDetId.sector());
    int rpc_trigger_phiB = (phiB == -10000) ? phiB : (int)round(phiB * m_dt_phiB_granularity);
    int rpc_quality = -1; // dummy for rpc
    int rpc_index = 0; // dummy for rpc
    return L1Phase2MuDTPhDigi(rpc_bx,
                    rpc_wheel,
                    trigger_sector,
                    rpc_station,
                    rpc_layer, //this would be the layer in the new dataformat
                    rpc_trigger_phi,
                    rpc_trigger_phiB,
                    rpc_quality,
                    rpc_index,
                    rpc_time,
                    -1, // no chi2 for RPC
                    rpc_flag);
}


double RPCIntegrator::getPhiBending(rpc_metaprimitive* rpc_hit_1, rpc_metaprimitive* rpc_hit_2) {
    // Adaptation of https://github.com/dtp2-tpg-am/cmssw/blob/AM_106X_dev/L1Trigger/DTPhase2Trigger/src/MuonPathAssociator.cc#L189
    DTChamberId DT_chamber(rpc_hit_1->rpc_id.ring(), rpc_hit_1->rpc_id.station(), rpc_hit_1->rpc_id.sector());
    LocalPoint lp_rpc_hit_1_dtconv = m_dtGeo->chamber(DT_chamber)->toLocal(rpc_hit_1->global_position);
    LocalPoint lp_rpc_hit_2_dtconv = m_dtGeo->chamber(DT_chamber)->toLocal(rpc_hit_2->global_position);
    double slope = (lp_rpc_hit_1_dtconv.x() - lp_rpc_hit_2_dtconv.x()) / distance_between_two_rpc_layers;
    double average_x = (lp_rpc_hit_1_dtconv.x() + lp_rpc_hit_2_dtconv.x()) / 2;
    GlobalPoint seg_middle_global = m_dtGeo->chamber(DT_chamber)->toGlobal(LocalPoint(average_x, 0., 0.)); // for station 1 and 2, z = 0
    double seg_phi = getPhi_DT_MP_conv(seg_middle_global.phi(), rpc_hit_1->rpc_id.sector());
    double psi = atan(slope);
    double phiB = hasPosRF_rpc(rpc_hit_1->rpc_id.ring(), rpc_hit_1->rpc_id.sector()) ? psi - seg_phi : -psi - seg_phi;
    return phiB;
}

int RPCIntegrator::getPhiInDTTPFormat(double rpc_global_phi, int rpcSector){
    double rpc_localDT_phi;
    rpc_localDT_phi = getPhi_DT_MP_conv(rpc_global_phi, rpcSector) * m_dt_phi_granularity;
    return (int)round(rpc_localDT_phi);
}

double RPCIntegrator::getPhi_DT_MP_conv(double rpc_global_phi, int rpcSector){
    // Adaptation of https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1TTwinMux/src/RPCtoDTTranslator.cc#L349
    if (rpcSector == 1) return rpc_global_phi;
    else {
        if (rpc_global_phi >= 0) return rpc_global_phi - (rpcSector - 1) * Geom::pi() / 6.;
        else return rpc_global_phi + (13 - rpcSector) * Geom::pi() / 6.;
    }
}

GlobalPoint RPCIntegrator::getRPCGlobalPosition(RPCDetId rpcId, const RPCRecHit& rpcIt) const{
  RPCDetId rpcid = RPCDetId(rpcId);
  const LocalPoint& rpc_lp = rpcIt.localPosition();
  const GlobalPoint& rpc_gp = m_rpcGeo->idToDet(rpcid)->surface().toGlobal(rpc_lp);

  return rpc_gp;
}

bool RPCIntegrator::hasPosRF_rpc(int wh, int sec) {
    return (wh > 0 || (wh == 0 && sec % 4 > 1));
}
