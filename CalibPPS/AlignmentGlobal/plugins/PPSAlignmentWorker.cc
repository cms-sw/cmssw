/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigurationRcd.h"

#include <map>
#include <string>
#include <cmath>
#include <memory>

#include "TH2D.h"
#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

class PPSAlignmentWorker : public DQMEDAnalyzer {
public:
  PPSAlignmentWorker(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // ------------ structures ------------
  struct SectorData {
    PPSAlignmentConfiguration::SectorConfig scfg_;

    // hit distributions
    std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_bef_sel;
    std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_mlt_sel;
    std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_aft_sel;

    // cut plots
    MonitorElement* h_q_cut_h_bef;
    MonitorElement* h_q_cut_h_aft;
    MonitorElement* h2_cut_h_bef;
    MonitorElement* h2_cut_h_aft;

    MonitorElement* h_q_cut_v_bef;
    MonitorElement* h_q_cut_v_aft;
    MonitorElement* h2_cut_v_bef;
    MonitorElement* h2_cut_v_aft;

    // near-far plots
    MonitorElement* p_x_diffFN_vs_x_N;
    MonitorElement* p_y_diffFN_vs_y_F;

    struct SlicePlots {
      MonitorElement* h_y;
      MonitorElement* h2_y_diffFN_vs_y;
      MonitorElement* p_y_diffFN_vs_y;

      SlicePlots();
      SlicePlots(DQMStore::IBooker& iBooker, const PPSAlignmentConfiguration& cfg, bool debug);
      void fill(const double y, const double yDiff, const bool debug);
    };

    std::map<unsigned int, SlicePlots> x_slice_plots_N, x_slice_plots_F;

    void init(DQMStore::IBooker& iBooker,
              const PPSAlignmentConfiguration& cfg,
              const PPSAlignmentConfiguration::SectorConfig& scfg,
              const std::string& rootDir,
              bool debug);

    unsigned int process(const CTPPSLocalTrackLiteCollection& tracks, const PPSAlignmentConfiguration& cfg, bool debug);
  };

  // ------------ member data ------------
  const edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esTokenBookHistograms_;
  const edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esTokenAnalyze_;

  const std::vector<edm::InputTag> tracksTags_;
  std::vector<edm::EDGetTokenT<CTPPSLocalTrackLiteCollection>> tracksTokens_;

  SectorData sectorData45_;
  SectorData sectorData56_;

  const std::string dqmDir_;
  const bool debug_;
};

// -------------------------------- DQMEDAnalyzer methods --------------------------------

PPSAlignmentWorker::PPSAlignmentWorker(const edm::ParameterSet& iConfig)
    : esTokenBookHistograms_(
          esConsumes<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd, edm::Transition::BeginRun>(
              edm::ESInputTag("", iConfig.getParameter<std::string>("label")))),
      esTokenAnalyze_(esConsumes<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("label")))),
      tracksTags_(iConfig.getParameter<std::vector<edm::InputTag>>("tracksTags")),
      dqmDir_(iConfig.getParameter<std::string>("dqm_dir")),
      debug_(iConfig.getParameter<bool>("debug")) {
  edm::LogInfo("PPSAlignmentWorker").log([&](auto& li) {
    li << "parameters:\n";
    li << "* label: " << iConfig.getParameter<std::string>("label") << "\n";
    li << "* tracksTags:\n";
    for (auto& tag : tracksTags_) {
      li << "    " << tag << ",\n";
    }
    li << "* dqm_dir: " << dqmDir_ << "\n";
    li << "* debug: " << std::boolalpha << debug_;
  });

  for (auto& tag : tracksTags_) {
    tracksTokens_.emplace_back(consumes<CTPPSLocalTrackLiteCollection>(tag));
  }
}

void PPSAlignmentWorker::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const& iSetup) {
  const auto& cfg = iSetup.getData(esTokenBookHistograms_);

  sectorData45_.init(iBooker, cfg, cfg.sectorConfig45(), dqmDir_ + "/worker", debug_);
  sectorData56_.init(iBooker, cfg, cfg.sectorConfig56(), dqmDir_ + "/worker", debug_);
}

void PPSAlignmentWorker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CTPPSLocalTrackLiteCollection tracks;
  bool foundProduct = false;

  for (unsigned int i = 0; i < tracksTokens_.size(); i++) {
    if (auto handle = iEvent.getHandle(tracksTokens_[i])) {
      tracks = *handle;
      foundProduct = true;
      edm::LogInfo("PPSAlignmentWorker") << "Found a product with " << tracksTags_[i];
      break;
    }
  }
  if (!foundProduct) {
    throw edm::Exception(edm::errors::ProductNotFound) << "Could not find a product with any of the selected labels.";
  }

  const auto& cfg = iSetup.getData(esTokenAnalyze_);

  sectorData45_.process(tracks, cfg, debug_);
  sectorData56_.process(tracks, cfg, debug_);
}

void PPSAlignmentWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "");
  desc.add<std::vector<edm::InputTag>>("tracksTags", {edm::InputTag("ctppsLocalTrackLiteProducer")});
  desc.add<std::string>("dqm_dir", "AlCaReco/PPSAlignment");
  desc.add<bool>("debug", false);

  descriptions.addWithDefaultLabel(desc);
}

// -------------------------------- SectorData and SlicePlots methods --------------------------------

PPSAlignmentWorker::SectorData::SlicePlots::SlicePlots() {}

PPSAlignmentWorker::SectorData::SlicePlots::SlicePlots(DQMStore::IBooker& iBooker,
                                                       const PPSAlignmentConfiguration& cfg,
                                                       bool debug) {
  h_y = iBooker.book1DD(
      "h_y", ";y", cfg.binning().slice_n_bins_x_, cfg.binning().slice_x_min_, cfg.binning().slice_x_max_);

  auto profilePtr = std::make_unique<TProfile>(
      "", ";y;y_{F} - y_{N}", cfg.binning().slice_n_bins_x_, cfg.binning().slice_x_min_, cfg.binning().slice_x_max_);
  p_y_diffFN_vs_y = iBooker.bookProfile("p_y_diffFN_vs_y", profilePtr.get());

  if (debug)
    h2_y_diffFN_vs_y = iBooker.book2DD("h2_y_diffFN_vs_y",
                                       ";y;y_{F} - y_{N}",
                                       cfg.binning().slice_n_bins_x_,
                                       cfg.binning().slice_x_min_,
                                       cfg.binning().slice_x_max_,
                                       cfg.binning().slice_n_bins_y_,
                                       cfg.binning().slice_y_min_,
                                       cfg.binning().slice_y_max_);
}

void PPSAlignmentWorker::SectorData::SlicePlots::fill(const double y, const double yDiff, const bool debug) {
  h_y->Fill(y);
  p_y_diffFN_vs_y->Fill(y, yDiff);
  if (debug)
    h2_y_diffFN_vs_y->Fill(y, yDiff);
}

void PPSAlignmentWorker::SectorData::init(DQMStore::IBooker& iBooker,
                                          const PPSAlignmentConfiguration& cfg,
                                          const PPSAlignmentConfiguration::SectorConfig& scfg,
                                          const std::string& rootDir,
                                          bool debug) {
  scfg_ = scfg;

  // binning
  const double bin_size_x = cfg.binning().bin_size_x_;
  const unsigned int n_bins_x = cfg.binning().n_bins_x_;

  const double pixel_x_offset = cfg.binning().pixel_x_offset_;

  const double x_min_pix = pixel_x_offset, x_max_pix = pixel_x_offset + n_bins_x * bin_size_x;
  const double x_min_str = 0., x_max_str = n_bins_x * bin_size_x;

  const unsigned int n_bins_y = cfg.binning().n_bins_y_;
  const double y_min = cfg.binning().y_min_, y_max = cfg.binning().y_max_;

  // hit distributions
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/before selection/" + scfg_.rp_N_.name_);
  m_h2_y_vs_x_bef_sel[scfg_.rp_N_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, n_bins_y, y_min, y_max);
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/before selection/" + scfg_.rp_F_.name_);
  m_h2_y_vs_x_bef_sel[scfg_.rp_F_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, n_bins_y, y_min, y_max);

  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/multiplicity selection/" + scfg_.rp_N_.name_);
  m_h2_y_vs_x_mlt_sel[scfg_.rp_N_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, n_bins_y, y_min, y_max);
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/multiplicity selection/" + scfg_.rp_F_.name_);
  m_h2_y_vs_x_mlt_sel[scfg_.rp_F_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, n_bins_y, y_min, y_max);

  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/after selection/" + scfg_.rp_N_.name_);
  m_h2_y_vs_x_aft_sel[scfg_.rp_N_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, n_bins_y, y_min, y_max);
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/after selection/" + scfg_.rp_F_.name_);
  m_h2_y_vs_x_aft_sel[scfg_.rp_F_.id_] =
      iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, n_bins_y, y_min, y_max);

  // cut plots
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/cuts/cut_h");
  h_q_cut_h_bef = iBooker.book1DD("h_q_cut_h_bef", ";cq_h", 400, -2., 2.);
  h_q_cut_h_aft = iBooker.book1DD("h_q_cut_h_aft", ";cq_h", 400, -2., 2.);
  h2_cut_h_bef =
      iBooker.book2DD("h2_cut_h_bef", ";x_up;x_dw", n_bins_x, x_min_str, x_max_str, n_bins_x, x_min_pix, x_max_pix);
  h2_cut_h_aft =
      iBooker.book2DD("h2_cut_h_aft", ";x_up;x_dw", n_bins_x, x_min_str, x_max_str, n_bins_x, x_min_pix, x_max_pix);

  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/cuts/cut_v");
  h_q_cut_v_bef = iBooker.book1DD("h_q_cut_v_bef", ";cq_v", 400, -2., 2.);
  h_q_cut_v_aft = iBooker.book1DD("h_q_cut_v_aft", ";cq_v", 400, -2., 2.);
  h2_cut_v_bef = iBooker.book2DD("h2_cut_v_bef", ";y_up;y_dw", n_bins_y, y_min, y_max, n_bins_y, y_min, y_max);
  h2_cut_v_aft = iBooker.book2DD("h2_cut_v_aft", ";y_up;y_dw", n_bins_y, y_min, y_max, n_bins_y, y_min, y_max);

  // near-far plots
  iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/near_far");

  auto profilePtr = std::make_unique<TProfile>("",
                                               ";x_{N};x_{F} - x_{N}",
                                               cfg.binning().diffFN_n_bins_x_,
                                               cfg.binning().diffFN_x_min_,
                                               cfg.binning().diffFN_x_max_);
  p_x_diffFN_vs_x_N = iBooker.bookProfile("p_x_diffFN_vs_x_N", profilePtr.get());

  // slice plots
  for (int i = 0; i < scfg_.rp_N_.x_slice_n_; i++) {
    const double x_min = scfg_.rp_N_.x_slice_min_ + i * scfg_.rp_N_.x_slice_w_;
    const double x_max = scfg_.rp_N_.x_slice_min_ + (i + 1) * scfg_.rp_N_.x_slice_w_;

    char buf[100];
    sprintf(buf, "%.1f-%.1f", x_min, x_max);
    std::replace(buf, buf + strlen(buf), '.', '_');  // replace . with _
    iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/near_far/x slices N/" + buf);
    x_slice_plots_N.insert({i, SlicePlots(iBooker, cfg, debug)});
  }

  for (int i = 0; i < scfg_.rp_F_.x_slice_n_; i++) {
    const double x_min = scfg_.rp_F_.x_slice_min_ + i * scfg_.rp_F_.x_slice_w_;
    const double x_max = scfg_.rp_F_.x_slice_min_ + (i + 1) * scfg_.rp_F_.x_slice_w_;

    char buf[100];
    sprintf(buf, "%.1f-%.1f", x_min, x_max);
    std::replace(buf, buf + strlen(buf), '.', '_');  // replace . with _
    iBooker.setCurrentFolder(rootDir + "/" + scfg_.name_ + "/near_far/x slices F/" + buf);
    x_slice_plots_F.insert({i, SlicePlots(iBooker, cfg, debug)});
  }
}

unsigned int PPSAlignmentWorker::SectorData::process(const CTPPSLocalTrackLiteCollection& tracks,
                                                     const PPSAlignmentConfiguration& cfg,
                                                     bool debug) {
  CTPPSLocalTrackLiteCollection tracksUp, tracksDw;

  for (const auto& tr : tracks) {
    CTPPSDetId rpId(tr.rpId());
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    if (rpDecId != scfg_.rp_N_.id_ && rpDecId != scfg_.rp_F_.id_)
      continue;

    // store the track in the right collection
    if (rpDecId == scfg_.rp_N_.id_)
      tracksUp.push_back(tr);
    if (rpDecId == scfg_.rp_F_.id_)
      tracksDw.push_back(tr);
  }

  // update plots before selection
  for (const auto& tr : tracksUp)
    m_h2_y_vs_x_bef_sel[scfg_.rp_N_.id_]->Fill(tr.x(), tr.y());

  for (const auto& tr : tracksDw)
    m_h2_y_vs_x_bef_sel[scfg_.rp_F_.id_]->Fill(tr.x(), tr.y());

  // skip crowded events (multiplicity selection)
  if (tracksUp.size() < cfg.minRPTracksSize() || tracksUp.size() > cfg.maxRPTracksSize())
    return 0;

  if (tracksDw.size() < cfg.minRPTracksSize() || tracksDw.size() > cfg.maxRPTracksSize())
    return 0;

  // update plots with multiplicity selection
  for (const auto& tr : tracksUp)
    m_h2_y_vs_x_mlt_sel[scfg_.rp_N_.id_]->Fill(tr.x(), tr.y());

  for (const auto& tr : tracksDw)
    m_h2_y_vs_x_mlt_sel[scfg_.rp_F_.id_]->Fill(tr.x(), tr.y());

  // do the selection
  unsigned int pairsSelected = 0;
  for (const auto& trUp : tracksUp) {
    for (const auto& trDw : tracksDw) {
      h2_cut_h_bef->Fill(trUp.x(), trDw.x());
      h2_cut_v_bef->Fill(trUp.y(), trDw.y());

      // horizontal cut
      const double cq_h = trDw.x() + scfg_.cut_h_a_ * trUp.x() + scfg_.cut_h_c_;
      h_q_cut_h_bef->Fill(cq_h);
      const bool cv_h = (std::fabs(cq_h) < cfg.n_si() * scfg_.cut_h_si_);

      // vertical cut
      const double cq_v = trDw.y() + scfg_.cut_v_a_ * trUp.y() + scfg_.cut_v_c_;
      h_q_cut_v_bef->Fill(cq_v);
      const bool cv_v = (std::fabs(cq_v) < cfg.n_si() * scfg_.cut_v_si_);

      bool cutsPassed = true;
      if (scfg_.cut_h_apply_)
        cutsPassed &= cv_h;
      if (scfg_.cut_v_apply_)
        cutsPassed &= cv_v;

      if (cutsPassed) {
        pairsSelected++;

        h_q_cut_h_aft->Fill(cq_h);
        h_q_cut_v_aft->Fill(cq_v);

        h2_cut_h_aft->Fill(trUp.x(), trDw.x());
        h2_cut_v_aft->Fill(trUp.y(), trDw.y());

        m_h2_y_vs_x_aft_sel[scfg_.rp_N_.id_]->Fill(trUp.x(), trUp.y());
        m_h2_y_vs_x_aft_sel[scfg_.rp_F_.id_]->Fill(trDw.x(), trDw.y());

        p_x_diffFN_vs_x_N->Fill(trUp.x(), trDw.x() - trUp.x());

        int idx = (trUp.x() - scfg_.rp_N_.x_slice_min_) / scfg_.rp_N_.x_slice_w_;
        if (idx >= 0 && idx < scfg_.rp_N_.x_slice_n_) {
          x_slice_plots_N[idx].fill(trUp.y(), trDw.y() - trUp.y(), debug);
        }

        idx = (trDw.x() - scfg_.rp_F_.x_slice_min_) / scfg_.rp_F_.x_slice_w_;
        if (idx >= 0 && idx < scfg_.rp_F_.x_slice_n_) {
          x_slice_plots_F[idx].fill(trDw.y(), trDw.y() - trUp.y(), debug);
        }
      }
    }
  }

  return pairsSelected;
}

DEFINE_FWK_MODULE(PPSAlignmentWorker);
