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
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

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

    //--- timing objects
    CTPPSTimingRecHit ctdm_rh;
    edm::Ptr<CTPPSTimingRecHit> ptr_ctdm_rh;
    edm::Wrapper<CTPPSTimingRecHit> wrp_ctdm_rh;
    std::vector<CTPPSTimingRecHit> vec_ctdm_rh;
    std::vector<edm::DetSet<CTPPSTimingRecHit> > vec_ds_ctdm_rh;
    edm::DetSet<CTPPSTimingRecHit> ds_ctdm_rh;
    edm::DetSetVector<CTPPSTimingRecHit> dsv_ctdm_rh;
    edm::Wrapper<edm::DetSetVector<CTPPSTimingRecHit> > wrp_dsv_ctdm_rh;
    edm::Wrapper<std::vector<CTPPSTimingRecHit> > wrp_vec_ctdm_rh;

    CTPPSDiamondRecHit ctd_rh;
    edm::Ptr<CTPPSDiamondRecHit> ptr_ctd_rh;
    edm::Wrapper<CTPPSDiamondRecHit> wrp_ctd_rh;
    std::vector<CTPPSDiamondRecHit> vec_ctd_rh;
    std::vector<edm::DetSet<CTPPSDiamondRecHit> > vec_ds_ctd_rh;
    edm::DetSet<CTPPSDiamondRecHit> ds_ctd_rh;
    edm::DetSetVector<CTPPSDiamondRecHit> dsv_ctd_rh;
    edm::Wrapper<edm::DetSetVector<CTPPSDiamondRecHit> > wrp_dsv_ctd_rh;
    edm::Wrapper<std::vector<CTPPSDiamondRecHit> > wrp_vec_ctd_rh;

    TotemTimingRecHit ttd_rh;
    edm::Ptr<TotemTimingRecHit> ptr_ttd_rh;
    edm::Wrapper<TotemTimingRecHit> wrp_ttd_rh;
    std::vector<TotemTimingRecHit> vec_ttd_rh;
    std::vector<edm::DetSet<TotemTimingRecHit> > vec_ds_ttd_rh;
    edm::DetSet<TotemTimingRecHit> ds_ttd_rh;
    edm::DetSetVector<TotemTimingRecHit> dsv_ttd_rh;
    edm::Wrapper<edm::DetSetVector<TotemTimingRecHit> > wrp_dsv_ttd_rh;
    edm::Wrapper<std::vector<TotemTimingRecHit> > wrp_vec_ttd_rh;

    CTPPSTimingLocalTrack ctdm_lt;
    edm::Ptr<CTPPSTimingLocalTrack> ptr_ctdm_lt;
    edm::Wrapper<CTPPSTimingLocalTrack> wrp_ctdm_lt;
    std::vector<CTPPSTimingLocalTrack> vec_ctdm_lt;
    edm::DetSet<CTPPSTimingLocalTrack> ds_ctdm_lt;
    std::vector<edm::DetSet<CTPPSTimingLocalTrack> > vec_ds_ctdm_lt;
    edm::Wrapper<std::vector<CTPPSTimingLocalTrack> > wrp_vec_ctdm_lt;
    edm::DetSetVector<CTPPSTimingLocalTrack> dsv_ctdm_lt;
    edm::Wrapper<edm::DetSetVector<CTPPSTimingLocalTrack> > wrp_dsv_ctdm_lt;

    CTPPSDiamondLocalTrack ctd_lt;
    edm::Ptr<CTPPSDiamondLocalTrack> ptr_ctd_lt;
    edm::Wrapper<CTPPSDiamondLocalTrack> wrp_ctd_lt;
    std::vector<CTPPSDiamondLocalTrack> vec_ctd_lt;
    edm::DetSet<CTPPSDiamondLocalTrack> ds_ctd_lt;
    std::vector<edm::DetSet<CTPPSDiamondLocalTrack> > vec_ds_ctd_lt;
    edm::Wrapper<std::vector<CTPPSDiamondLocalTrack> > wrp_vec_ctd_lt;
    edm::DetSetVector<CTPPSDiamondLocalTrack> dsv_ctd_lt;
    edm::Wrapper<edm::DetSetVector<CTPPSDiamondLocalTrack> > wrp_dsv_ctd_lt;

    //--- pixel objects

    CTPPSPixelCluster rpcl;
    edm::DetSet<CTPPSPixelCluster> dsrpcl;
    std::vector<CTPPSPixelCluster> svrpcl;
    std::vector<edm::DetSet<CTPPSPixelCluster> > svdsrpcl;
    edm::DetSetVector<CTPPSPixelCluster> dsvrpcl;
    edm::Wrapper<edm::DetSetVector<CTPPSPixelCluster> > wdsvrpcl;

    CTPPSPixelRecHit rprh;
    edm::DetSet<CTPPSPixelRecHit> dsrprh;
    std::vector<CTPPSPixelRecHit> svrprh;
    std::vector<edm::DetSet<CTPPSPixelRecHit> > svdsrprh;
    edm::DetSetVector<CTPPSPixelRecHit> dsvrprh;
    edm::Wrapper<edm::DetSetVector<CTPPSPixelRecHit> > wdsvrprh;

    CTPPSPixelLocalTrack rplt;
    edm::DetSet<CTPPSPixelLocalTrack> dsrplt;
    std::vector<CTPPSPixelLocalTrack> svrplt;
    std::vector<edm::DetSet<CTPPSPixelLocalTrack> > svdsrplt;
    edm::DetSetVector<CTPPSPixelLocalTrack> dsvrplt;
    edm::Wrapper<edm::DetSetVector<CTPPSPixelLocalTrack> > wdsvrplt;
    edm::DetSetVector<CTPPSPixelFittedRecHit> dsvrplcfrh;
    edm::Wrapper<edm::DetSetVector<CTPPSPixelFittedRecHit>> wdsvrplcfrh;
    std::vector<edm::DetSet<CTPPSPixelFittedRecHit> > vdsrpltfrh;
    std::vector<CTPPSPixelFittedRecHit> vrpltfrh;
    CTPPSPixelFittedRecHit pfrh;
    edm::Wrapper<CTPPSPixelFittedRecHit> wpfrh;
    //--- common objects

    CTPPSLocalTrackLite cltl;
    std::vector<CTPPSLocalTrackLite> v_cltl;
    edm::Wrapper<CTPPSLocalTrackLite> w_cltl;
    edm::Wrapper<std::vector<CTPPSLocalTrackLite>> w_v_cltl;
  };
}
