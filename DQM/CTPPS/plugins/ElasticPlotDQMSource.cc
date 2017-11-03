/****************************************************************************
*
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

// TODO: clean
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
 
class ElasticPlotDQMSource: public DQMEDAnalyzer
{
  public:
    ElasticPlotDQMSource(const edm::ParameterSet& ps);
    ~ElasticPlotDQMSource() override;
  
  protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;

    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

    void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;

    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;

    void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  private:
    static constexpr double ls_duration = 23.357; // s
    static constexpr unsigned int ls_min = 0;
    static constexpr unsigned int ls_max = 2000; // little more than 12h

    unsigned int verbosity;

    edm::EDGetTokenT< edm::DetSetVector<TotemRPRecHit> > tokenRecHit;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPUVPattern> > tokenUVPattern;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > tokenLocalTrack;

    /// plots related to one (anti)diagonal
    struct DiagonalPlots
    {
      // track correlation in vertical RPs
      MonitorElement *h2_track_corr_vert=nullptr;

      // y distributions in a RP (1st index) with another RP (2nd index) in coincidence
      std::vector<std::vector<MonitorElement *>> v_h_y;

      // XY hit maps in a give RP (vector index) under these conditions
      //   4rp: all 4 diagonal RPs have a track
      //   2rp: diagonal RPs in 220-fr have a track
      std::vector<MonitorElement* > v_h2_y_vs_x_dgn_4rp;
      std::vector<MonitorElement *> v_h2_y_vs_x_dgn_2rp;

      // event rates vs. time
      MonitorElement *h_rate_vs_time_dgn_4rp=nullptr;
      MonitorElement *h_rate_vs_time_dgn_2rp=nullptr;

      DiagonalPlots() {}

      DiagonalPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, DiagonalPlots> diagonalPlots;

    /// plots related to one RP
    struct PotPlots
    {
      // event rates vs. time
      //  suff = singal sufficient to reconstruct track
      //  track = track is reconstructed
      //  unresolved = suff && !track
      MonitorElement *h_rate_vs_time_suff=nullptr;
      MonitorElement *h_rate_vs_time_track=nullptr;
      MonitorElement *h_rate_vs_time_unresolved=nullptr;

      PotPlots() {}
      PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PotPlots> potPlots;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

ElasticPlotDQMSource::DiagonalPlots::DiagonalPlots(DQMStore::IBooker &ibooker, int id)
{
  bool top45 = id & 2;
  bool top56 = id & 1;
  bool diag = (top45 != top56);

  char name[50];
  sprintf(name, "%s 45%s - 56%s",
    (diag) ? "diagonal" : "antidiagonal",
    (top45) ? "top" : "bot",
    (top56) ? "top" : "bot"
  );

  ibooker.setCurrentFolder(string("CTPPS/TrackingStrip/") + name);

  // TODO
}

//----------------------------------------------------------------------------------------------------

ElasticPlotDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  string path;
  TotemRPDetId(id).rpName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).rpName(title, TotemRPDetId::nFull);

  h_rate_vs_time_suff = ibooker.book1D("rate - suff", title+";lumi section", ls_max-ls_min+1, -0.5+ls_min, +0.5+ls_max);
  h_rate_vs_time_track = ibooker.book1D("rate - track", title+";lumi section", ls_max-ls_min+1, -0.5+ls_min, +0.5+ls_max);
  h_rate_vs_time_unresolved = ibooker.book1D("rate - unresolved", title+";lumi section", ls_max-ls_min+1, -0.5+ls_min, +0.5+ls_max);
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

ElasticPlotDQMSource::ElasticPlotDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenRecHit = consumes< edm::DetSetVector<TotemRPRecHit> >(ps.getParameter<edm::InputTag>("tagRecHit"));
  tokenUVPattern = consumes< DetSetVector<TotemRPUVPattern> >(ps.getParameter<edm::InputTag>("tagUVPattern"));
  tokenLocalTrack = consumes< DetSetVector<TotemRPLocalTrack> >(ps.getParameter<edm::InputTag>("tagLocalTrack"));
}

//----------------------------------------------------------------------------------------------------

ElasticPlotDQMSource::~ElasticPlotDQMSource()
{
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::dqmBeginRun(edm::Run const &run, edm::EventSetup const &)
{
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &)
{
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  // initialize diagonal plots
  diagonalPlots[1] = DiagonalPlots(ibooker, 1);  // 45 bot - 56 top
  diagonalPlots[2] = DiagonalPlots(ibooker, 2);  // 45 top - 45 bot

  // loop over arms
  for (unsigned int arm = 0; arm < 2; arm++)
  {
    // loop over stations
    for (unsigned int st = 0; st < 3; st += 2)
    {
      // loop over RPs
      for (unsigned int rp = 0; rp < 6; ++rp)
      {
        // skip horizontals - irrelevant for elastics
        if (rp == 2 || rp == 3)
          continue;

        // skip "nr" units - not equipped
        if (rp <= 2)
          continue;

        TotemRPDetId rpId(arm, st, rp);
        potPlots[rpId] = PotPlots(ibooker, rpId);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  // get event setup data
  ESHandle<CTPPSGeometry> geometry;
  eventSetup.get<VeryForwardRealGeometryRecord>().get(geometry);

  // get event data
  Handle< DetSetVector<TotemRPRecHit> > hits;
  event.getByToken(tokenRecHit, hits);

  Handle<DetSetVector<TotemRPUVPattern>> patterns;
  event.getByToken(tokenUVPattern, patterns);

  Handle< DetSetVector<TotemRPLocalTrack> > tracks;
  event.getByToken(tokenLocalTrack, tracks);

  // check validity
  bool valid = true;
  valid &= hits.isValid();
  valid &= patterns.isValid();
  valid &= tracks.isValid();

  if (!valid)
  {
    if (verbosity)
    {
      LogProblem("ElasticPlotDQMSource") <<
        "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
        << "    hits.isValid = " << hits.isValid() << "\n"
        << "    patterns.isValid = " << patterns.isValid() << "\n"
        << "    tracks.isValid = " << tracks.isValid();
    }

    return;
  }

  //------------------------------
  // categorise RP data
  map<unsigned int, unsigned int> rp_planes_u_too_full, rp_planes_v_too_full;
  map<unsigned int, bool> rp_pat_suff, rp_has_track;

  for (const auto &ds : *hits)
  {
    TotemRPDetId detId(ds.detId());
    CTPPSDetId rpId = detId.getRPId();

    if (ds.size() > 5)
    {
      if (detId.isStripsCoordinateUDirection())
        rp_planes_u_too_full[rpId]++;
      else
        rp_planes_v_too_full[rpId]++;
    }
  }

  for (auto &ds : *patterns)
  {
    CTPPSDetId rpId(ds.detId());

    // count U and V patterns
    unsigned int n_pat_u = 0, n_pat_v = 0;
    for (auto &p : ds)
    {
      if (! p.getFittable())
        continue;

      if (p.getProjection() == TotemRPUVPattern::projU)
        n_pat_u++;

      if (p.getProjection() == TotemRPUVPattern::projV)
        n_pat_v++;
    }

    rp_pat_suff[rpId] = (n_pat_u > 0 || rp_planes_u_too_full[rpId] >= 3) && (n_pat_v > 0 || rp_planes_v_too_full[rpId] >= 3);
  }

  for (auto &ds : *tracks)
  {
    CTPPSDetId rpId(ds.detId());

    bool trackPresent = false;
    for (auto &ft : ds)
    {
      if (ft.isValid())
      {
        trackPresent = true;
        break;
      }
    }

    rp_has_track[rpId] = trackPresent;
  } 

  //------------------------------
  // diagonal plots

  // TODO: remove
  printf("%u\n", event.luminosityBlock());

  // TODO
  
  //------------------------------
  // pot plots

  for (const auto &p : rp_pat_suff)
  {
    const auto &rpId = p.first;
    auto pp_it = potPlots.find(rpId);
    if (pp_it == potPlots.end())
      continue;
    auto &pp = pp_it->second;

    const auto &pat_suff = rp_pat_suff[rpId];
    const auto &has_track = rp_has_track[rpId];

    if (pat_suff)
      pp.h_rate_vs_time_suff->Fill(event.luminosityBlock(), 1./ls_duration);
    if (has_track)
      pp.h_rate_vs_time_track->Fill(event.luminosityBlock(), 1./ls_duration);
    if (pat_suff && !has_track)
      pp.h_rate_vs_time_unresolved->Fill(event.luminosityBlock(), 1./ls_duration);
  }
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//----------------------------------------------------------------------------------------------------

void ElasticPlotDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(ElasticPlotDQMSource);
