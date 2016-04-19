#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include <vector>

namespace {
  namespace {
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
  }
}
