//-------------------------------------------------
//
//   Class: RPCHitCleaner
//
//   RPCHitCleaner
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//   modifications:   G Karathanasis     U Athens
//--------------------------------------------------

#include <iostream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <map>

#include "L1Trigger/L1TTwinMux/interface/RPCHitCleaner.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/L1TTwinMux/interface/RPCtoDTTranslator.h"

using namespace std;


RPCHitCleaner::RPCHitCleaner(RPCDigiCollection const& inrpcDigis):
  m_inrpcDigis {inrpcDigis}
{
}

namespace {
  constexpr int max_rpc_bx = 3; 
  constexpr int min_rpc_bx = -3;

  //Need to shift the index so that index 0
  // corresponds to min_rpc_bx
  class BxToStrips {
  public:
    BxToStrips(): m_strips{} {} //zero initializes

    static bool outOfRange(int iBX) {
      return (iBX > max_rpc_bx or iBX < min_rpc_bx) ;
    }

    int& operator[](int iBX) {
      return m_strips[iBX-min_rpc_bx];
    }

    size_t size() const { return m_strips.size(); }
  private:
    std::array<int,max_rpc_bx-min_rpc_bx+1> m_strips;
  };
}

void RPCHitCleaner::run(const edm::EventSetup& c) {

  std::map<detId_Ext, int> hits;
  vector<int> vcluster_size;
  std::map<RPCDetId, int> bx_hits;

  int cluster_size = 0;
  int cluster_id = -1;
  int itr=0;

  for( auto chamber = m_inrpcDigis.begin(); chamber != m_inrpcDigis.end(); ++chamber ){
        RPCDetId detid = (*chamber).first;
        int strip_n1 = -10000;
        int bx_n1 = -10000;
        if(detid.region()!=0 ) continue; //Region = 0 Barrel
        for( auto digi = (*chamber).second.first ; digi != (*chamber).second.second; ++digi ) {
             if(fabs(digi->bx())>3 ) continue;
             ///Create cluster ids and store their size
             //if((digi->strip()-1!=strip_n1) || digi->bx()!=bx_n1){
             if( abs(digi->strip()-strip_n1)!=1 || digi->bx()!=bx_n1){
               if(itr!=0)vcluster_size.push_back(cluster_size);
                 cluster_size = 0;
                 cluster_id++;
                 }
                itr++;
                cluster_size++;
                ///hit belongs to cluster with clusterid
                detId_Ext tmp{detid,digi->bx(),digi->strip()};
                hits[tmp] = cluster_id;
                ///strip of i-1
                strip_n1 = digi->strip();
                bx_n1 = digi->bx();
                }///for digicout
    }///for chamber
  vcluster_size.push_back(cluster_size);

  for( auto chamber = m_inrpcDigis.begin(); chamber != m_inrpcDigis.end(); ++chamber ){
        RPCDetId detid = (*chamber).first;
        if(detid.region()!=0 ) continue; //Region = 0 Barrel
        BxToStrips strips;
        int cluster_n1 = -10;
        bx_hits[detid] = 10;
        //Keep cluster with min bx in a roll
        for( auto digi = (*chamber).second.first ; digi != (*chamber).second.second; ++digi ) {
            if(BxToStrips::outOfRange(digi->bx()) ) continue;
            //int cluster_id =  hits[(detid.ring()+2)][(detid.station()-1)][(detid.sector()-1)][(detid.layer()-1)][(digi->bx()+2)][detid.roll()-1][digi->strip()];
             detId_Ext tmp{detid,digi->bx(),digi->strip()};
             int cluster_id = hits[tmp];
             ///Remove clusters with size>=4
             if( vcluster_size[cluster_id] >=4 ) continue;
             if(bx_hits[detid]>digi->bx())
                bx_hits[detid] = digi->bx();
               }

         for( auto digi = (*chamber).second.first ; digi != (*chamber).second.second; ++digi ) {
              if(fabs(digi->bx())>3 ) continue;
              detId_Ext tmp{detid,digi->bx(),digi->strip()};
              int cluster_id = hits[tmp];
              ///Remove clusters with size>=4
              if( vcluster_size[cluster_id] >=4 ) continue;
              ///keep only one bx per st/sec/wheel/layer
              if(digi->bx()!=bx_hits[detid] ) continue;
              ///Count strips in a cluster
              if(cluster_n1 != cluster_id) {strips[digi->bx()] = {0}; }
              strips[digi->bx()] ++ ;
              cluster_n1 = cluster_id;

              if( vcluster_size[cluster_id] ==3 && strips[digi->bx()]!=2) continue;
               ///Keep clusters with size=2. Calculate and store the mean phi in RPCtoDTTranslator
              RPCDigi digi_out(digi->strip(), digi->bx());
              m_outrpcDigis.insertDigi(detid, digi_out);
              }///for digicout
    }///for chamber
}
