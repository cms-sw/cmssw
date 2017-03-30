#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include <vector>

namespace DataFormats_CTPPSReco {
  struct dictionary {

    //--- strips objects

    TotemRPRecHit rp_reco_hit;
    edm::DetSet<TotemRPRecHit> ds_rp_reco_hit;
    edm::DetSetVector<TotemRPRecHit> dsv_rp_reco_hit;
    std::vector<edm::DetSet<TotemRPRecHit> > sv_dsw_rp_reco_hit;
    edm::Wrapper<edm::DetSetVector<TotemRPRecHit> > w_dsv_rp_reco_hit;
    std::vector<TotemRPRecHit> sv_rp_reco_hit;
    std::vector<const TotemRPRecHit*> sv_cp_rp_reco_hit;
    
    TotemRPCluster dc;
    edm::DetSet<TotemRPCluster> dsdc;
    std::vector<TotemRPCluster> svdc;
    std::vector<edm::DetSet<TotemRPCluster> > svdsdc;
    edm::DetSetVector<TotemRPCluster> dsvdc;
    edm::Wrapper<edm::DetSetVector<TotemRPCluster> > wdsvdc;

    TotemRPUVPattern pat;
    edm::DetSetVector<TotemRPUVPattern> dsv_pat;
    edm::Wrapper<edm::DetSetVector<TotemRPUVPattern>> w_dsv_pat;

    TotemRPLocalTrack ft;
    edm::DetSetVector<TotemRPLocalTrack> dsv_ft;
    edm::Wrapper<edm::DetSetVector<TotemRPLocalTrack>> w_dsv_ft;
    edm::DetSetVector<TotemRPLocalTrack::FittedRecHit> dsv_ft_frh;
    edm::Wrapper<edm::DetSetVector<TotemRPLocalTrack::FittedRecHit>> w_dsv_ft_frh;
    std::vector<edm::DetSet<TotemRPLocalTrack::FittedRecHit> > v_ds_ft_frh;
    std::vector<TotemRPLocalTrack::FittedRecHit> v_ft_frh;

    //--- diamonds objects

    CTPPSDiamondRecHit ctd_rh;
    edm::Ptr<CTPPSDiamondRecHit> ptr_ctd_rh;
    edm::Wrapper<CTPPSDiamondRecHit> wrp_ctd_rh;
    std::vector<CTPPSDiamondRecHit> vec_rh;
    std::vector< edm::DetSet<CTPPSDiamondRecHit> > vec_ds_rh;
    edm::DetSet<CTPPSDiamondRecHit> ds_rh;
    edm::DetSetVector<CTPPSDiamondRecHit> dsv_ctd_rh;
    edm::Wrapper< edm::DetSetVector<CTPPSDiamondRecHit> > wrp_dsv_ctd_rh;

    //--- common objects

    CTPPSLocalTrackLite cltl;
    std::vector<CTPPSLocalTrackLite> v_cltl;
    edm::Wrapper<CTPPSLocalTrackLite> w_cltl;
    edm::Wrapper<std::vector<CTPPSLocalTrackLite>> w_v_cltl;
  };
}
