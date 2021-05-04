/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class TotemRPDQMSource : public DQMEDAnalyzer {
public:
  TotemRPDQMSource(const edm::ParameterSet &ps);
  ~TotemRPDQMSource() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  unsigned int verbosity;

  edm::EDGetTokenT<edm::DetSetVector<TotemVFATStatus>> tokenStatus;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPDigi>> tokenDigi;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPCluster>> tokenCluster;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPRecHit>> tokenRecHit;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPUVPattern>> tokenUVPattern;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> tokenLocalTrack;

  /// plots related to one RP
  struct PotPlots {
    MonitorElement *vfat_problem = nullptr, *vfat_missing = nullptr, *vfat_ec_bc_error = nullptr,
                   *vfat_corruption = nullptr;

    MonitorElement *activity = nullptr, *activity_u = nullptr, *activity_v = nullptr;
    MonitorElement *activity_per_bx = nullptr, *activity_per_bx_short = nullptr;
    MonitorElement *hit_plane_hist = nullptr;
    MonitorElement *patterns_u = nullptr, *patterns_v = nullptr;
    MonitorElement *h_planes_fit_u = nullptr, *h_planes_fit_v = nullptr;
    MonitorElement *event_category = nullptr;
    MonitorElement *trackHitsCumulativeHist = nullptr;
    MonitorElement *track_u_profile = nullptr, *track_v_profile = nullptr;
    MonitorElement *triggerSectorUVCorrelation_all = nullptr, *triggerSectorUVCorrelation_mult2 = nullptr,
                   *triggerSectorUVCorrelation_track = nullptr;

    PotPlots() {}
    PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::map<unsigned int, PotPlots> potPlots;

  /// plots related to one RP plane
  struct PlanePlots {
    MonitorElement *digi_profile_cumulative = nullptr;
    MonitorElement *cluster_profile_cumulative = nullptr;
    MonitorElement *hit_multiplicity = nullptr;
    MonitorElement *cluster_size = nullptr;
    MonitorElement *efficiency_num = nullptr, *efficiency_den = nullptr;

    PlanePlots() {}
    PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::map<unsigned int, PlanePlots> planePlots;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  string path;
  TotemRPDetId(id).rpName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).rpName(title, TotemRPDetId::nFull);

  vfat_problem = ibooker.book2D("vfats with any problem", title + ";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_missing = ibooker.book2D("vfats missing", title + ";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_ec_bc_error =
      ibooker.book2D("vfats with EC or BC error", title + ";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_corruption =
      ibooker.book2D("vfats with data corruption", title + ";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);

  activity = ibooker.book1D("active planes", title + ";number of active planes", 11, -0.5, 10.5);
  activity_u = ibooker.book1D("active planes U", title + ";number of active U planes", 11, -0.5, 10.5);
  activity_v = ibooker.book1D("active planes V", title + ";number of active V planes", 11, -0.5, 10.5);

  activity_per_bx = ibooker.book1D("activity per BX", title + ";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_bx_short = ibooker.book1D("activity per BX (short)", title + ";Event.BX", 102, -1.5, 100. + 0.5);

  hit_plane_hist =
      ibooker.book2D("activity in planes (2D)", title + ";plane number;strip number", 10, -0.5, 9.5, 32, -0.5, 511.5);

  patterns_u = ibooker.book1D("recognized patterns U", title + ";number of recognized U patterns", 11, -0.5, 10.5);
  patterns_v = ibooker.book1D("recognized patterns V", title + ";number of recognized V patterns", 11, -0.5, 10.5);

  h_planes_fit_u =
      ibooker.book1D("planes contributing to fit U", title + ";number of planes contributing to U fit", 6, -0.5, 5.5);
  h_planes_fit_v =
      ibooker.book1D("planes contributing to fit V", title + ";number of planes contributing to V fit", 6, -0.5, 5.5);

  event_category = ibooker.book1D("event category", title + ";event category", 5, -0.5, 4.5);
  TH1F *event_category_h = event_category->getTH1F();
  event_category_h->GetXaxis()->SetBinLabel(1, "empty");
  event_category_h->GetXaxis()->SetBinLabel(2, "insufficient");
  event_category_h->GetXaxis()->SetBinLabel(3, "single-track");
  event_category_h->GetXaxis()->SetBinLabel(4, "multi-track");
  event_category_h->GetXaxis()->SetBinLabel(5, "shower");

  trackHitsCumulativeHist =
      ibooker.book2D("track XY profile", title + ";x   (mm);y   (mm)", 100, -18., +18., 100, -18., +18.);

  track_u_profile = ibooker.book1D("track profile U", title + "; U   (mm)", 512, -256 * 66E-3, +256 * 66E-3);
  track_v_profile = ibooker.book1D("track profile V", title + "; V   (mm)", 512, -256 * 66E-3, +256 * 66E-3);

  triggerSectorUVCorrelation_all = ibooker.book2D(
      "trigger sector UV correlation (no cond)", title + ";V sector;U sector", 16, -0.5, 15.5, 16, -0.5, 15.5);
  triggerSectorUVCorrelation_mult2 = ibooker.book2D(
      "trigger sector UV correlation (max mult 2)", title + ";V sector;U sector", 16, -0.5, 15.5, 16, -0.5, 15.5);
  triggerSectorUVCorrelation_track = ibooker.book2D(
      "trigger sector UV correlation (track)", title + ";V sector;U sector", 16, -0.5, 15.5, 16, -0.5, 15.5);
}

//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id) {
  string path;
  TotemRPDetId(id).planeName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).planeName(title, TotemRPDetId::nFull);

  digi_profile_cumulative = ibooker.book1D("digi profile", title + ";strip number", 512, -0.5, 511.5);
  cluster_profile_cumulative = ibooker.book1D("cluster profile", title + ";cluster center", 1024, -0.25, 511.75);
  hit_multiplicity = ibooker.book1D("hit multiplicity", title + ";hits/detector/event", 6, -0.5, 5.5);
  cluster_size = ibooker.book1D("cluster size", title + ";hits per cluster", 5, 0.5, 5.5);

  efficiency_num = ibooker.book1D("efficiency num", title + ";track position   (mm)", 30, -15., 0.);
  efficiency_den = ibooker.book1D("efficiency den", title + ";track position   (mm)", 30, -15., 0.);
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::TotemRPDQMSource(const edm::ParameterSet &ps)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)) {
  tokenStatus = consumes<DetSetVector<TotemVFATStatus>>(ps.getParameter<edm::InputTag>("tagStatus"));

  tokenDigi = consumes<DetSetVector<TotemRPDigi>>(ps.getParameter<edm::InputTag>("tagDigi"));
  tokenCluster = consumes<edm::DetSetVector<TotemRPCluster>>(ps.getParameter<edm::InputTag>("tagCluster"));
  tokenRecHit = consumes<edm::DetSetVector<TotemRPRecHit>>(ps.getParameter<edm::InputTag>("tagRecHit"));
  tokenUVPattern = consumes<DetSetVector<TotemRPUVPattern>>(ps.getParameter<edm::InputTag>("tagUVPattern"));
  tokenLocalTrack = consumes<DetSetVector<TotemRPLocalTrack>>(ps.getParameter<edm::InputTag>("tagLocalTrack"));
}

//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::~TotemRPDQMSource() {}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  // loop over arms
  for (unsigned int arm : {0, 1}) {
    // loop over RPs
    for (unsigned int st_rp : {2, 3, 4, 5, 24, 25}) {
      const unsigned int st = st_rp / 10;
      const unsigned int rp = st_rp % 10;

      TotemRPDetId rpId(arm, st, rp);
      potPlots[rpId] = PotPlots(ibooker, rpId);

      // loop over planes
      for (unsigned int pl = 0; pl < 10; ++pl) {
        TotemRPDetId plId(arm, st, rp, pl);
        planePlots[plId] = PlanePlots(ibooker, plId);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  // get event setup data
  ESHandle<CTPPSGeometry> geometry;
  eventSetup.get<VeryForwardRealGeometryRecord>().get(geometry);

  // get event data
  Handle<DetSetVector<TotemVFATStatus>> status;
  event.getByToken(tokenStatus, status);

  Handle<DetSetVector<TotemRPDigi>> digi;
  event.getByToken(tokenDigi, digi);

  Handle<DetSetVector<TotemRPCluster>> digCluster;
  event.getByToken(tokenCluster, digCluster);

  Handle<DetSetVector<TotemRPRecHit>> hits;
  event.getByToken(tokenRecHit, hits);

  Handle<DetSetVector<TotemRPUVPattern>> patterns;
  event.getByToken(tokenUVPattern, patterns);

  Handle<DetSetVector<TotemRPLocalTrack>> tracks;
  event.getByToken(tokenLocalTrack, tracks);

  // check validity
  bool valid = true;
  valid &= status.isValid();
  valid &= digi.isValid();
  valid &= digCluster.isValid();
  valid &= hits.isValid();
  valid &= patterns.isValid();
  valid &= tracks.isValid();

  if (!valid) {
    if (verbosity) {
      LogProblem("TotemRPDQMSource")
          << "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
          << "    status.isValid = " << status.isValid() << "\n"
          << "    digi.isValid = " << digi.isValid() << "\n"
          << "    digCluster.isValid = " << digCluster.isValid() << "\n"
          << "    hits.isValid = " << hits.isValid() << "\n"
          << "    patterns.isValid = " << patterns.isValid() << "\n"
          << "    tracks.isValid = " << tracks.isValid();
    }

    return;
  }

  //------------------------------
  // Status Plots

  for (auto &ds : *status) {
    TotemRPDetId detId(ds.detId());
    unsigned int plNum = detId.plane();
    CTPPSDetId rpId = detId.rpId();

    auto it = potPlots.find(rpId);
    if (it == potPlots.end())
      continue;
    auto &plots = it->second;

    for (auto &s : ds) {
      if (s.isMissing()) {
        plots.vfat_problem->Fill(plNum, s.chipPosition());
        plots.vfat_missing->Fill(plNum, s.chipPosition());
      }

      if (s.isECProgressError() || s.isBCProgressError()) {
        plots.vfat_problem->Fill(plNum, s.chipPosition());
        plots.vfat_ec_bc_error->Fill(plNum, s.chipPosition());
      }

      if (s.isIDMismatch() || s.isFootprintError() || s.isCRCError()) {
        plots.vfat_problem->Fill(plNum, s.chipPosition());
        plots.vfat_corruption->Fill(plNum, s.chipPosition());
      }
    }
  }

  //------------------------------
  // Plane Plots

  // digi profile cumulative
  for (DetSetVector<TotemRPDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it) {
    TotemRPDetId detId(it->detId());

    auto plIt = planePlots.find(detId);
    if (plIt == planePlots.end())
      continue;
    auto &plots = plIt->second;

    for (DetSet<TotemRPDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      plots.digi_profile_cumulative->Fill(dit->stripNumber());
  }

  // cluster plots
  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++) {
    TotemRPDetId detId(it->detId());

    auto plIt = planePlots.find(detId);
    if (plIt == planePlots.end())
      continue;
    auto &plots = plIt->second;

    // hit multiplicity
    plots.hit_multiplicity->Fill(it->size());

    for (DetSet<TotemRPCluster>::const_iterator dit = it->begin(); dit != it->end(); ++dit) {
      // profile cumulative
      plots.cluster_profile_cumulative->Fill(dit->centerStripPosition());

      // cluster size
      plots.cluster_size->Fill(dit->numberOfStrips());
    }
  }

  // plane efficiency plots
  for (auto &ds : *tracks) {
    CTPPSDetId rpId(ds.detId());

    for (auto &ft : ds) {
      if (!ft.isValid())
        continue;

      double rp_z = geometry->rpTranslation(rpId).z();

      for (unsigned int plNum = 0; plNum < 10; ++plNum) {
        TotemRPDetId plId = rpId;
        plId.setPlane(plNum);

        auto plIt = planePlots.find(plId);
        if (plIt == planePlots.end())
          continue;
        auto &plots = plIt->second;

        double ft_z = ft.z0();
        double ft_x = ft.x0() + ft.tx() * (ft_z - rp_z);
        double ft_y = ft.y0() + ft.ty() * (ft_z - rp_z);

        double ft_v = geometry->globalToLocal(plId, CTPPSGeometry::Vector(ft_x, ft_y, ft_z)).y();

        bool hasMatchingHit = false;
        const auto &hit_ds_it = hits->find(plId);
        if (hit_ds_it != hits->end()) {
          for (const auto &h : *hit_ds_it) {
            bool match = (fabs(ft_v - h.position()) < 2. * 0.066);
            if (match) {
              hasMatchingHit = true;
              break;
            }
          }
        }

        plots.efficiency_den->Fill(ft_v);
        if (hasMatchingHit)
          plots.efficiency_num->Fill(ft_v);
      }
    }
  }

  //------------------------------
  // Roman Pots Plots

  // determine active planes (from RecHits and VFATStatus)
  map<unsigned int, set<unsigned int>> planes;
  map<unsigned int, set<unsigned int>> planes_u;
  map<unsigned int, set<unsigned int>> planes_v;
  for (const auto &ds : *hits) {
    if (ds.empty())
      continue;

    TotemRPDetId detId(ds.detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.rpId();

    planes[rpId].insert(planeNum);
    if (detId.isStripsCoordinateUDirection())
      planes_u[rpId].insert(planeNum);
    else
      planes_v[rpId].insert(planeNum);
  }

  for (const auto &ds : *status) {
    bool activity = false;
    for (const auto &s : ds) {
      if (s.isNumberOfClustersSpecified() && s.numberOfClusters() > 0) {
        activity = true;
        break;
      }
    }

    if (!activity)
      continue;

    TotemRPDetId detId(ds.detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.rpId();

    planes[rpId].insert(planeNum);
    if (detId.isStripsCoordinateUDirection())
      planes_u[rpId].insert(planeNum);
    else
      planes_v[rpId].insert(planeNum);
  }

  // plane activity histogram
  for (std::map<unsigned int, PotPlots>::iterator it = potPlots.begin(); it != potPlots.end(); it++) {
    it->second.activity->Fill(planes[it->first].size());
    it->second.activity_u->Fill(planes_u[it->first].size());
    it->second.activity_v->Fill(planes_v[it->first].size());

    if (planes[it->first].size() >= 6) {
      it->second.activity_per_bx->Fill(event.bunchCrossing());
      it->second.activity_per_bx_short->Fill(event.bunchCrossing());
    }
  }

  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++) {
    TotemRPDetId detId(it->detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.rpId();

    auto plIt = potPlots.find(rpId);
    if (plIt == potPlots.end())
      continue;
    auto &plots = plIt->second;

    for (DetSet<TotemRPCluster>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      plots.hit_plane_hist->Fill(planeNum, dit->centerStripPosition());
  }

  // recognized pattern histograms
  for (auto &ds : *patterns) {
    CTPPSDetId rpId(ds.detId());

    auto plIt = potPlots.find(rpId);
    if (plIt == potPlots.end())
      continue;
    auto &plots = plIt->second;

    // count U and V patterns
    unsigned int u = 0, v = 0;
    for (auto &p : ds) {
      if (!p.fittable())
        continue;

      if (p.projection() == TotemRPUVPattern::projU)
        u++;

      if (p.projection() == TotemRPUVPattern::projV)
        v++;
    }

    plots.patterns_u->Fill(u);
    plots.patterns_v->Fill(v);
  }

  // event-category histogram
  for (auto &it : potPlots) {
    TotemRPDetId rpId(it.first);
    auto &pp = it.second;

    // process hit data for this plot
    unsigned int pl_u = planes_u[rpId].size();
    unsigned int pl_v = planes_v[rpId].size();

    // process pattern data for this pot
    const auto &rp_pat_it = patterns->find(rpId);

    unsigned int pat_u = 0, pat_v = 0;
    if (rp_pat_it != patterns->end()) {
      for (auto &p : *rp_pat_it) {
        if (!p.fittable())
          continue;

        if (p.projection() == TotemRPUVPattern::projU)
          pat_u++;

        if (p.projection() == TotemRPUVPattern::projV)
          pat_v++;
      }
    }

    // determine category
    signed int category = -1;

    if (pl_u == 0 && pl_v == 0)
      category = 0;  // empty

    if (category == -1 && pat_u + pat_v <= 1) {
      if (pl_u + pl_v < 6)
        category = 1;  // insuff
      else
        category = 4;  // shower
    }

    if (pat_u == 1 && pat_v == 1)
      category = 2;  // 1-track

    if (category == -1)
      category = 3;  // multi-track

    pp.event_category->Fill(category);
  }

  // RP track-fit plots
  set<unsigned int> rps_with_tracks;

  for (auto &ds : *tracks) {
    CTPPSDetId rpId(ds.detId());

    rps_with_tracks.insert(rpId);

    auto plIt = potPlots.find(rpId);
    if (plIt == potPlots.end())
      continue;
    auto &plots = plIt->second;

    for (auto &ft : ds) {
      if (!ft.isValid())
        continue;

      // number of planes contributing to (valid) fits
      unsigned int n_pl_in_fit_u = 0, n_pl_in_fit_v = 0;
      for (auto &hds : ft.hits()) {
        TotemRPDetId plId(hds.detId());
        bool uProj = plId.isStripsCoordinateUDirection();

        for (auto &h : hds) {
          h.position();  // just to keep compiler silent
          if (uProj)
            n_pl_in_fit_u++;
          else
            n_pl_in_fit_v++;
        }
      }

      plots.h_planes_fit_u->Fill(n_pl_in_fit_u);
      plots.h_planes_fit_v->Fill(n_pl_in_fit_v);

      // mean position of U and V planes
      TotemRPDetId plId_V(rpId);
      plId_V.setPlane(0);
      TotemRPDetId plId_U(rpId);
      plId_U.setPlane(1);

      double rp_x = (geometry->sensor(plId_V)->translation().x() + geometry->sensor(plId_U)->translation().x()) / 2.;
      double rp_y = (geometry->sensor(plId_V)->translation().y() + geometry->sensor(plId_U)->translation().y()) / 2.;

      // mean read-out direction of U and V planes
      const auto &rod_U = geometry->localToGlobalDirection(plId_U, CTPPSGeometry::Vector(0., 1., 0.));
      const auto &rod_V = geometry->localToGlobalDirection(plId_V, CTPPSGeometry::Vector(0., 1., 0.));

      double x = ft.x0() - rp_x;
      double y = ft.y0() - rp_y;

      plots.trackHitsCumulativeHist->Fill(x, y);

      double U = x * rod_U.x() + y * rod_U.y();
      double V = x * rod_V.x() + y * rod_V.y();

      plots.track_u_profile->Fill(U);
      plots.track_v_profile->Fill(V);
    }
  }

  // restore trigger-sector map from digis
  map<unsigned int, map<unsigned int, map<unsigned int, unsigned int>>>
      triggerSectorMap;  // [rpId, U/V flag, sector] --> number of planes
  for (const auto &dp : *digi) {
    TotemRPDetId plId(dp.detId());
    CTPPSDetId rpId = plId.rpId();
    unsigned int uvFlag = (plId.isStripsCoordinateUDirection()) ? 0 : 1;

    set<unsigned int> sectors;
    for (const auto &d : dp) {
      unsigned int sector = d.stripNumber() / 32;
      sectors.insert(sector);
    }

    for (const auto &sector : sectors)
      triggerSectorMap[rpId][uvFlag][sector]++;
  }

  for (auto &rpp : triggerSectorMap) {
    const unsigned int rpId = rpp.first;

    // trigger sector is counted as active if at least 3 planes report activity

    set<unsigned int> triggerSectorsU;
    for (const auto sp : rpp.second[0]) {
      if (sp.second >= 3)
        triggerSectorsU.insert(sp.first);
    }

    set<unsigned int> triggerSectorsV;
    for (const auto sp : rpp.second[1]) {
      if (sp.second >= 3)
        triggerSectorsV.insert(sp.first);
    }

    auto plIt = potPlots.find(rpId);
    if (plIt == potPlots.end())
      continue;
    auto &plots = plIt->second;

    const bool high_mult = (triggerSectorsU.size() > 2 && triggerSectorsV.size() > 2);

    const bool has_track = (rps_with_tracks.find(rpId) != rps_with_tracks.end());

    for (const auto &secU : triggerSectorsU) {
      for (const auto &secV : triggerSectorsV) {
        plots.triggerSectorUVCorrelation_all->Fill(secV, secU);

        if (!high_mult)
          plots.triggerSectorUVCorrelation_mult2->Fill(secV, secU);

        if (has_track)
          plots.triggerSectorUVCorrelation_track->Fill(secV, secU);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPDQMSource);
