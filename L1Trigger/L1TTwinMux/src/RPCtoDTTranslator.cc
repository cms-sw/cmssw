//-------------------------------------------------
//
//   Class: RPCtoDTTranslator
//
//   RPCtoDTTranslator
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//   modifications: G Karathanasis U Athens
//--------------------------------------------------

#include <iostream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <map>

#include "L1Trigger/L1TTwinMux/interface/RPCHitCleaner.h"
#include "L1Trigger/L1TTwinMux/interface/RPCtoDTTranslator.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/L1TCommon/interface/BitShift.h"

using namespace std;

RPCtoDTTranslator::RPCtoDTTranslator(RPCDigiCollection const& inrpcDigis) : m_rpcDigis{inrpcDigis} {}

namespace {
  constexpr int max_rpc_bx = 2;
  constexpr int min_rpc_bx = -2;

  struct rpc_hit {
    int bx;
    int station;
    int sector;
    int wheel;
    RPCDetId detid;
    int strip;
    int roll;
    int layer;
    rpc_hit(int pbx, int pstation, int psector, int pwheel, RPCDetId pdet, int pstrip, int proll, int player)
        : bx(pbx),
          station(pstation),
          sector(psector),
          wheel(pwheel),
          detid(pdet),
          strip(pstrip),
          roll(proll),
          layer(player) {}
  };

  //Need to shift the index so that index 0
  // corresponds to min_rpc_bx
  class BxToHit {
  public:
    BxToHit() : m_hits{} {}  //zero initializes

    static bool outOfRange(int iBX) { return (iBX > max_rpc_bx or iBX < min_rpc_bx); }

    int& operator[](int iBX) { return m_hits[iBX - min_rpc_bx]; }

    size_t size() const { return m_hits.size(); }

  private:
    std::array<int, max_rpc_bx - min_rpc_bx + 1> m_hits;
  };
}  // namespace

void RPCtoDTTranslator::run(const RPCGeometry& rpcGeometry) {
  std::vector<L1MuDTChambPhDigi> l1ttma_out;
  std::vector<L1MuDTChambPhDigi> l1ttma_hits_out;

  std::vector<rpc_hit> vrpc_hit_layer1, vrpc_hit_layer2, vrpc_hit_st3, vrpc_hit_st4;

  ///Init structues
  for (auto chamber = m_rpcDigis.begin(); chamber != m_rpcDigis.end(); ++chamber) {
    RPCDetId detid = (*chamber).first;
    for (auto digi = (*chamber).second.first; digi != (*chamber).second.second; ++digi) {
      if (detid.region() != 0)
        continue;  //Region = 0 Barrel
      if (BxToHit::outOfRange(digi->bx()))
        continue;
      if (detid.layer() == 1)
        vrpc_hit_layer1.emplace_back(digi->bx(),
                                     detid.station(),
                                     detid.sector(),
                                     detid.ring(),
                                     detid,
                                     digi->strip(),
                                     detid.roll(),
                                     detid.layer());
      if (detid.station() == 3)
        vrpc_hit_st3.emplace_back(digi->bx(),
                                  detid.station(),
                                  detid.sector(),
                                  detid.ring(),
                                  detid,
                                  digi->strip(),
                                  detid.roll(),
                                  detid.layer());
      if (detid.layer() == 2)
        vrpc_hit_layer2.emplace_back(digi->bx(),
                                     detid.station(),
                                     detid.sector(),
                                     detid.ring(),
                                     detid,
                                     digi->strip(),
                                     detid.roll(),
                                     detid.layer());
      if (detid.station() == 4)
        vrpc_hit_st4.emplace_back(digi->bx(),
                                  detid.station(),
                                  detid.sector(),
                                  detid.ring(),
                                  detid,
                                  digi->strip(),
                                  detid.roll(),
                                  detid.layer());
    }
  }  ///for chamber

  vector<int> vcluster_size;
  int cluster_id = -1;
  int itr = 0;
  //      int hits[5][4][12][2][5][3][100]= {{{{{{{0}}}}}}};
  std::map<RPCHitCleaner::detId_Ext, int> hits;
  int cluster_size = 0;
  for (auto chamber = m_rpcDigis.begin(); chamber != m_rpcDigis.end(); ++chamber) {
    RPCDetId detid = (*chamber).first;
    int strip_n1 = -10000;
    int bx_n1 = -10000;
    if (detid.region() != 0)
      continue;  //Region = 0 Barrel
    for (auto digi = (*chamber).second.first; digi != (*chamber).second.second; ++digi) {
      if (fabs(digi->bx()) > 3)
        continue;
      //Create cluster ids and store their size
      //if((digi->strip()+1!=strip_n1)|| digi->bx()!=bx_n1){
      if (abs(digi->strip() - strip_n1) != 1 || digi->bx() != bx_n1) {
        if (itr != 0)
          vcluster_size.push_back(cluster_size);
        cluster_size = 0;
        cluster_id++;
      }
      itr++;
      cluster_size++;
      ///hit belongs to cluster with clusterid
      //hits[(detid.ring()+2)][(detid.station()-1)][(detid.sector()-1)][(detid.layer()-1)][(digi->bx()+2)][detid.roll()-1][digi->strip()]= cluster_id ;
      RPCHitCleaner::detId_Ext tmp{detid, digi->bx(), digi->strip()};
      hits[tmp] = cluster_id;
      ///strip of i-1
      strip_n1 = digi->strip();
      bx_n1 = digi->bx();
    }
  }  ///for chamber
  vcluster_size.push_back(cluster_size);

  for (int wh = -2; wh <= 2; wh++) {
    for (int sec = 1; sec <= 12; sec++) {
      for (int st = 1; st <= 4; st++) {
        int rpcbx = 0;
        std::vector<int> delta_phib;
        bool found_hits = false;
        std::vector<int> rpc2dt_phi, rpc2dt_phib;
        ///Loop over all combinations of layer 1 and 2.
        int itr1 = 0;
        for (unsigned int l1 = 0; l1 < vrpc_hit_layer1.size(); l1++) {
          RPCHitCleaner::detId_Ext tmp{vrpc_hit_layer1[l1].detid, vrpc_hit_layer1[l1].bx, vrpc_hit_layer1[l1].strip};
          int id = hits[tmp];
          int phi1 = radialAngle(vrpc_hit_layer1[l1].detid, rpcGeometry, vrpc_hit_layer1[l1].strip);
          if (vcluster_size[id] == 2 && itr1 == 0) {
            itr1++;
            continue;
          }
          if (vcluster_size[id] == 2 && itr1 == 1) {
            itr1 = 0;
            phi1 = phi1 + (radialAngle(vrpc_hit_layer1[l1 - 1].detid, rpcGeometry, vrpc_hit_layer1[l1 - 1].strip));
            phi1 /= 2;
          }
          int itr2 = 0;
          for (unsigned int l2 = 0; l2 < vrpc_hit_layer2.size(); l2++) {
            if (vrpc_hit_layer1[l1].station != st || vrpc_hit_layer2[l2].station != st)
              continue;
            if (vrpc_hit_layer1[l1].sector != sec || vrpc_hit_layer2[l2].sector != sec)
              continue;
            if (vrpc_hit_layer1[l1].wheel != wh || vrpc_hit_layer2[l2].wheel != wh)
              continue;
            if (vrpc_hit_layer1[l1].bx != vrpc_hit_layer2[l2].bx)
              continue;
            RPCHitCleaner::detId_Ext tmp{vrpc_hit_layer2[l2].detid, vrpc_hit_layer2[l2].bx, vrpc_hit_layer2[l2].strip};
            int id = hits[tmp];

            if (vcluster_size[id] == 2 && itr2 == 0) {
              itr2++;
              continue;
            }

            //int phi1 = radialAngle(vrpc_hit_layer1[l1].detid, c, vrpc_hit_layer1[l1].strip) ;
            int phi2 = radialAngle(vrpc_hit_layer2[l2].detid, rpcGeometry, vrpc_hit_layer2[l2].strip);
            if (vcluster_size[id] == 2 && itr2 == 1) {
              itr2 = 0;
              phi2 = phi2 + (radialAngle(vrpc_hit_layer2[l2 - 1].detid, rpcGeometry, vrpc_hit_layer2[l2 - 1].strip));
              phi2 /= 2;
            }
            int average = l1t::bitShift(((phi1 + phi2) / 2), 2);  //10-bit->12-bit
            rpc2dt_phi.push_back(average);                        //Convert and store to 12-bit
            //int xin = localX(vrpc_hit_layer1[l1].detid, c, vrpc_hit_layer1[l1].strip);
            //int xout = localX(vrpc_hit_layer2[l2].detid, c, vrpc_hit_layer2[l2].strip);
            //cout<<(phi1<<2)<<"   "<<l1<<"   "<<vrpc_hit_layer1[l1].station<<endl;
            //cout<<(phi2<<2)<<"   "<<l1<<"   "<<vrpc_hit_layer1[l1].station<<endl;
            int xin = localXX(l1t::bitShift(phi1, 2), 1, vrpc_hit_layer1[l1].station);
            int xout = localXX(l1t::bitShift(phi2, 2), 2, vrpc_hit_layer2[l2].station);
            if (vcluster_size[id] == 2 && itr2 == 1) {
              int phi1_n1 = radialAngle(vrpc_hit_layer1[l1 - 1].detid, rpcGeometry, vrpc_hit_layer1[l1 - 1].strip);
              int phi2_n1 = radialAngle(vrpc_hit_layer2[l2 - 1].detid, rpcGeometry, vrpc_hit_layer2[l2 - 1].strip);
              xin += localXX(l1t::bitShift(phi1_n1, 2), 1, vrpc_hit_layer1[l1].station);
              xout += localXX(l1t::bitShift(phi2_n1, 2), 2, vrpc_hit_layer2[l2].station);
              xin /= 2;
              xout /= 2;
            }
            //cout<<">>"<<xin<<"   "<<xout<<endl;
            int phi_b = bendingAngle(xin, xout, average);
            //cout<<"phib   "<<phi_b<<endl;
            rpc2dt_phib.push_back(phi_b);
            //delta_phib to find the highest pt primitve
            delta_phib.push_back(abs(phi_b));
            found_hits = true;
            rpcbx = vrpc_hit_layer1[l1].bx;
          }
        }
        if (found_hits) {
          //cout<<"found_hits"<<endl;
          int min_index = std::distance(delta_phib.begin(), std::min_element(delta_phib.begin(), delta_phib.end())) + 0;
          //cout<<min_index<<endl;
          l1ttma_out.emplace_back(rpcbx, wh, sec - 1, st, rpc2dt_phi[min_index], rpc2dt_phib[min_index], 3, 0, 0, 2);
        }
        ///Use ts2tag variable to store N rpchits for the same st/wheel/sec
        BxToHit hit;
        itr1 = 0;
        for (unsigned int l1 = 0; l1 < vrpc_hit_layer1.size(); l1++) {
          if (vrpc_hit_layer1[l1].station != st || st > 2 || vrpc_hit_layer1[l1].sector != sec ||
              vrpc_hit_layer1[l1].wheel != wh)
            continue;
          //int id = hits[vrpc_hit_layer1[l1].wheel+2][(vrpc_hit_layer1[l1].station-1)][(vrpc_hit_layer1[l1].sector-1)][(vrpc_hit_layer1[l1].layer-1)][(vrpc_hit_layer1[l1].bx+2)][vrpc_hit_layer1[l1].roll-1][vrpc_hit_layer1[l1].strip];
          RPCHitCleaner::detId_Ext tmp{vrpc_hit_layer1[l1].detid, vrpc_hit_layer1[l1].bx, vrpc_hit_layer1[l1].strip};
          int id = hits[tmp];
          if (vcluster_size[id] == 2 && itr1 == 0) {
            itr1++;
            continue;
          }
          int phi2 = radialAngle(vrpc_hit_layer1[l1].detid, rpcGeometry, vrpc_hit_layer1[l1].strip);
          phi2 = l1t::bitShift(phi2, 2);
          if (vcluster_size[id] == 2 && itr1 == 1) {
            itr1 = 0;
            phi2 =
                phi2 + l1t::bitShift(
                           radialAngle(vrpc_hit_layer1[l1 - 1].detid, rpcGeometry, vrpc_hit_layer1[l1 - 1].strip), 2);
            phi2 /= 2;
          }

          l1ttma_hits_out.emplace_back(
              vrpc_hit_layer1[l1].bx, wh, sec - 1, st, phi2, 0, 3, hit[vrpc_hit_layer1[l1].bx], 0, 2);
          hit[vrpc_hit_layer1[l1].bx]++;
        }
        itr1 = 0;
        for (unsigned int l2 = 0; l2 < vrpc_hit_layer2.size(); l2++) {
          if (vrpc_hit_layer2[l2].station != st || st > 2 || vrpc_hit_layer2[l2].sector != sec ||
              vrpc_hit_layer2[l2].wheel != wh)
            continue;
          RPCHitCleaner::detId_Ext tmp{vrpc_hit_layer2[l2].detid, vrpc_hit_layer2[l2].bx, vrpc_hit_layer2[l2].strip};
          int id = hits[tmp];
          //              int id = hits[vrpc_hit_layer2[l2].wheel+2][(vrpc_hit_layer2[l2].station-1)][(vrpc_hit_layer2[l2].sector-1)][(vrpc_hit_layer2[l2].layer-1)][(vrpc_hit_layer2[l2].bx+2)][vrpc_hit_layer2[l2].roll-1][vrpc_hit_layer2[l2].strip];
          if (vcluster_size[id] == 2 && itr1 == 0) {
            itr1++;
            continue;
          }
          int phi2 = radialAngle(vrpc_hit_layer2[l2].detid, rpcGeometry, vrpc_hit_layer2[l2].strip);
          phi2 = l1t::bitShift(phi2, 2);
          if (vcluster_size[id] == 2 && itr1 == 1) {
            itr1 = 0;
            phi2 =
                phi2 + l1t::bitShift(
                           radialAngle(vrpc_hit_layer2[l2 - 1].detid, rpcGeometry, vrpc_hit_layer2[l2 - 1].strip), 2);
            phi2 /= 2;
          }
          l1ttma_hits_out.emplace_back(
              vrpc_hit_layer2[l2].bx, wh, sec - 1, st, phi2, 0, 3, hit[vrpc_hit_layer2[l2].bx], 0, 2);
          hit[vrpc_hit_layer2[l2].bx]++;
        }
        itr1 = 0;

        for (unsigned int l1 = 0; l1 < vrpc_hit_st3.size(); l1++) {
          if (st != 3 || vrpc_hit_st3[l1].station != 3 || vrpc_hit_st3[l1].wheel != wh ||
              vrpc_hit_st3[l1].sector != sec)
            continue;
          RPCHitCleaner::detId_Ext tmp{vrpc_hit_st3[l1].detid, vrpc_hit_st3[l1].bx, vrpc_hit_st3[l1].strip};
          int id = hits[tmp];
          //int id = hits[vrpc_hit_st3[l1].wheel+2][(vrpc_hit_st3[l1].station-1)][(vrpc_hit_st3[l1].sector-1)][(vrpc_hit_st3[l1].layer-1)][(vrpc_hit_st3[l1].bx+2)][vrpc_hit_st3[l1].roll-1][vrpc_hit_st3[l1].strip];
          if (vcluster_size[id] == 2 && itr1 == 0) {
            itr1++;
            continue;
          }
          int phi2 = radialAngle(vrpc_hit_st3[l1].detid, rpcGeometry, vrpc_hit_st3[l1].strip);
          phi2 = l1t::bitShift(phi2, 2);
          if (vcluster_size[id] == 2 && itr1 == 1) {
            itr1 = 0;
            phi2 = phi2 +
                   l1t::bitShift(radialAngle(vrpc_hit_st3[l1 - 1].detid, rpcGeometry, vrpc_hit_st3[l1 - 1].strip), 2);
            phi2 /= 2;
          }
          l1ttma_hits_out.emplace_back(
              vrpc_hit_st3[l1].bx, wh, sec - 1, st, phi2, 0, 3, hit[vrpc_hit_st3[l1].bx], 0, 2);
          hit[vrpc_hit_st3[l1].bx]++;
        }
        itr1 = 0;

        for (unsigned int l1 = 0; l1 < vrpc_hit_st4.size(); l1++) {
          if (st != 4 || vrpc_hit_st4[l1].station != 4 || vrpc_hit_st4[l1].wheel != wh ||
              vrpc_hit_st4[l1].sector != sec)
            continue;
          //int id = hits[vrpc_hit_st4[l1].wheel+2][(vrpc_hit_st4[l1].station-1)][(vrpc_hit_st4[l1].sector-1)][(vrpc_hit_st4[l1].layer-1)][(vrpc_hit_st4[l1].bx+2)][vrpc_hit_st4[l1].roll-1][vrpc_hit_st4[l1].strip];
          RPCHitCleaner::detId_Ext tmp{vrpc_hit_st4[l1].detid, vrpc_hit_st4[l1].bx, vrpc_hit_st4[l1].strip};
          int id = hits[tmp];
          if (vcluster_size[id] == 2 && itr1 == 0) {
            itr1++;
            continue;
          }
          int phi2 = radialAngle(vrpc_hit_st4[l1].detid, rpcGeometry, vrpc_hit_st4[l1].strip);
          phi2 = l1t::bitShift(phi2, 2);
          if (vcluster_size[id] == 2 && itr1 == 1) {
            itr1 = 0;
            phi2 = phi2 +
                   l1t::bitShift(radialAngle(vrpc_hit_st4[l1 - 1].detid, rpcGeometry, vrpc_hit_st4[l1 - 1].strip), 2);
            phi2 /= 2;
          }
          l1ttma_hits_out.emplace_back(
              vrpc_hit_st4[l1].bx, wh, sec - 1, st, phi2, 0, 3, hit[vrpc_hit_st4[l1].bx], 0, 2);
          hit[vrpc_hit_st4[l1].bx]++;
          //l1ttma_out.push_back(rpc2dt_out);

          //break;
        }
      }
    }
  }
  ///Container to store RPC->DT for RPC only (only in stations 1 and 2 (2 layers->phib))
  m_rpcdt_translated.setContainer(l1ttma_out);
  ///Container to store RPC->DT for Bx correction
  m_rpchitsdt_translated.setContainer(l1ttma_hits_out);
}

///function - will be replaced by LUTs(?)
int RPCtoDTTranslator::radialAngle(RPCDetId detid, const RPCGeometry& rpcGeometry, int strip) {
  int radialAngle;
  // from phiGlobal to radialAngle of the primitive in sector sec in [1..12] <- RPC scheme

  const RPCRoll* roll = rpcGeometry.roll(detid);
  GlobalPoint stripPosition = roll->toGlobal(roll->centreOfStrip(strip));

  double globalphi = stripPosition.phi();
  int sector = (roll->id()).sector();
  if (sector == 1)
    radialAngle = int(globalphi * 1024);
  else {
    if (globalphi >= 0)
      radialAngle = int((globalphi - (sector - 1) * Geom::pi() / 6.) * 1024);
    else
      radialAngle = int((globalphi + (13 - sector) * Geom::pi() / 6.) * 1024);
  }
  return radialAngle;
}

///function - will be replaced by LUTs(?)
int RPCtoDTTranslator::localX(RPCDetId detid, const RPCGeometry& rpcGeometry, int strip) {
  const RPCRoll* roll = rpcGeometry.roll(detid);

  ///Orientaion of RPCs
  GlobalPoint p1cmPRG = roll->toGlobal(LocalPoint(1, 0, 0));
  GlobalPoint m1cmPRG = roll->toGlobal(LocalPoint(-1, 0, 0));
  float phip1cm = p1cmPRG.phi();
  float phim1cm = m1cmPRG.phi();
  int direction = (phip1cm - phim1cm) / abs(phip1cm - phim1cm);
  ///---Orientaion

  return direction * roll->centreOfStrip(strip).x();
}

int RPCtoDTTranslator::bendingAngle(int xin, int xout, int phi) {
  // use chamber size and max angle in hw units 512
  int atanv = (int)(atan((xout - xin) / 34.6) * 512);
  if (atanv > 512)
    return 512;
  int rvalue = atanv - phi / 8;
  return rvalue;
}

int RPCtoDTTranslator::localXX(int phi, int layer, int station) {
  //R[stat][layer] - radius of rpc station/layer from center of CMS
  double R[2][2] = {{410.0, 444.8}, {492.7, 527.3}};
  double rvalue = R[station - 1][layer - 1] * tan(phi / 4096.);
  return rvalue;
}
