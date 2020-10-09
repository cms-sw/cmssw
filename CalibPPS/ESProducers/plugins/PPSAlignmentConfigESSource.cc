/****************************************************************************
 *
 *  CalibPPS/ESProducers/plugins/PPSAlignmentConfigESSource.cc
 * 
 *  Description: Constructs PPSAlignmentConfig instance
 *
 *  Authors:
 *  - Jan Kašpar
 *  - Mateusz Kocot
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <queue>

#include "TF1.h"
#include "TProfile.h"
#include "TFile.h"
#include "TKey.h"
#include "TSystemFile.h"

//---------------------------------------------------------------------------------------------

class PPSAlignmentConfigESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PPSAlignmentConfigESSource(const edm::ParameterSet &iConfig);

  std::unique_ptr<PPSAlignmentConfig> produce(const PPSAlignmentConfigRcd &);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  int fitProfile(TProfile *p, double x_mean, double x_rms, double &sl, double &sl_unc);
  TDirectory *findDirectoryWithName(TDirectory *dir, std::string searchName);
  std::vector<PointErrors> buildVectorFromDirectory(TDirectory *dir, const RPConfig &rpd);

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                      const edm::IOVSyncValue &iosv,
                      edm::ValidityInterval &oValidity) override;

  bool debug;

  std::vector<std::string> sequence;
  std::string resultsDir;

  SectorConfig sectorConfig45, sectorConfig56;

  double x_ali_sh_step;

  double y_mode_sys_unc;
  double chiSqThreshold;
  double y_mode_unc_max_valid;
  double y_mode_max_valid;

  unsigned int maxRPTracksSize;
  double n_si;

  std::map<unsigned int, std::vector<PointErrors>> matchingReferencePoints;
  std::map<unsigned int, SelectionRange> matchingShiftRanges;

  std::map<unsigned int, SelectionRange> alignment_x_meth_o_ranges;
  unsigned int fitProfileMinBinEntries;
  unsigned int fitProfileMinNReasonable;
  unsigned int methOGraphMinN;
  double methOUncFitRange;

  std::map<unsigned int, SelectionRange> alignment_x_relative_ranges;
  unsigned int nearFarMinEntries;

  std::map<unsigned int, SelectionRange> alignment_y_ranges;
  unsigned int modeGraphMinN;
  unsigned int multSelProjYMinEntries;

  Binning binning;

  std::string label;
};

//---------------------------------------------------------------------------------------------

PPSAlignmentConfigESSource::PPSAlignmentConfigESSource(const edm::ParameterSet &iConfig) {
  label = iConfig.getParameter<std::string>("label");

  debug = iConfig.getParameter<bool>("debug");
  TFile *debugFile = nullptr;
  if (debug) {
    debugFile = new TFile(("debug_producer_" + (label.empty() ? "test" : label) + ".root").c_str(), "recreate");
  }

  sequence = iConfig.getParameter<std::vector<std::string>>("sequence");
  resultsDir = iConfig.getParameter<std::string>("results_dir");

  sectorConfig45.name_ = "sector 45";

  sectorConfig45.rp_N_.position_ = "N";
  sectorConfig45.rp_F_.position_ = "F";

  sectorConfig56.name_ = "sector 56";

  sectorConfig56.rp_N_.position_ = "N";
  sectorConfig56.rp_F_.position_ = "F";

  for (std::string sectorName : {"sector_45", "sector_56"}) {
    const auto &sps = iConfig.getParameter<edm::ParameterSet>(sectorName);
    SectorConfig *sc;
    if (sectorName == "sector_45")
      sc = &sectorConfig45;
    else
      sc = &sectorConfig56;

    for (std::string rpName : {"rp_N", "rp_F"}) {
      const auto &rpps = sps.getParameter<edm::ParameterSet>(rpName);
      RPConfig *rc;
      if (rpName == "rp_N")
        rc = &sc->rp_N_;
      else
        rc = &sc->rp_F_;

      rc->name_ = rpps.getParameter<std::string>("name");
      rc->id_ = rpps.getParameter<int>("id");

      rc->slope_ = rpps.getParameter<double>("slope");
      rc->sh_x_ = rpps.getParameter<double>("sh_x");

      rc->x_min_fit_mode_ = rpps.getParameter<double>("x_min_fit_mode");
      rc->x_max_fit_mode_ = rpps.getParameter<double>("x_max_fit_mode");
      rc->y_max_fit_mode_ = rpps.getParameter<double>("y_max_fit_mode");
      rc->y_cen_add_ = rpps.getParameter<double>("y_cen_add");
      rc->y_width_mult_ = rpps.getParameter<double>("y_width_mult");

      rc->x_slice_min_ = rpps.getParameter<double>("x_slice_min");
      rc->x_slice_w_ = rpps.getParameter<double>("x_slice_w");
      rc->x_slice_n_ = std::ceil((rpps.getParameter<double>("x_slice_max") - rc->x_slice_min_) / rc->x_slice_w_);
    }

    sc->slope_ = sps.getParameter<double>("slope");

    sc->cut_h_apply_ = sps.getParameter<bool>("cut_h_apply");
    sc->cut_h_a_ = sps.getParameter<double>("cut_h_a");
    sc->cut_h_c_ = sps.getParameter<double>("cut_h_c");
    sc->cut_h_si_ = sps.getParameter<double>("cut_h_si");

    sc->cut_v_apply_ = sps.getParameter<bool>("cut_v_apply");
    sc->cut_v_a_ = sps.getParameter<double>("cut_v_a");
    sc->cut_v_c_ = sps.getParameter<double>("cut_v_c");
    sc->cut_v_si_ = sps.getParameter<double>("cut_v_si");
  }

  std::map<unsigned int, std::string> rpTags = {{sectorConfig45.rp_F_.id_, "rp_L_F"},
                                                {sectorConfig45.rp_N_.id_, "rp_L_N"},
                                                {sectorConfig56.rp_N_.id_, "rp_R_N"},
                                                {sectorConfig56.rp_F_.id_, "rp_R_F"}};

  std::map<unsigned int, std::string> sectorNames = {{sectorConfig45.rp_F_.id_, sectorConfig45.name_},
                                                     {sectorConfig45.rp_N_.id_, sectorConfig45.name_},
                                                     {sectorConfig56.rp_N_.id_, sectorConfig56.name_},
                                                     {sectorConfig56.rp_F_.id_, sectorConfig56.name_}};

  std::map<unsigned int, const RPConfig *> rpConfigs = {{sectorConfig45.rp_F_.id_, &sectorConfig45.rp_F_},
                                                        {sectorConfig45.rp_N_.id_, &sectorConfig45.rp_N_},
                                                        {sectorConfig56.rp_N_.id_, &sectorConfig56.rp_N_},
                                                        {sectorConfig56.rp_F_.id_, &sectorConfig56.rp_F_}};

  x_ali_sh_step = iConfig.getParameter<double>("x_ali_sh_step");

  y_mode_sys_unc = iConfig.getParameter<double>("y_mode_sys_unc");
  chiSqThreshold = iConfig.getParameter<double>("chiSqThreshold");
  y_mode_unc_max_valid = iConfig.getParameter<double>("y_mode_unc_max_valid");
  y_mode_max_valid = iConfig.getParameter<double>("y_mode_max_valid");

  maxRPTracksSize = iConfig.getParameter<unsigned int>("max_RP_tracks_size");
  n_si = iConfig.getParameter<double>("n_si");

  const auto &c_axo = iConfig.getParameter<edm::ParameterSet>("x_alignment_meth_o");
  for (const auto &p : rpTags) {
    const auto &ps = c_axo.getParameter<edm::ParameterSet>(p.second);
    alignment_x_meth_o_ranges[p.first] = {ps.getParameter<double>("x_min"), ps.getParameter<double>("x_max")};
  }
  fitProfileMinBinEntries = c_axo.getParameter<unsigned int>("fit_profile_min_bin_entries");
  fitProfileMinNReasonable = c_axo.getParameter<unsigned int>("fit_profile_min_N_reasonable");
  methOGraphMinN = c_axo.getParameter<unsigned int>("meth_o_graph_min_N");
  methOUncFitRange = c_axo.getParameter<double>("meth_o_unc_fit_range");

  const auto &c_m = iConfig.getParameter<edm::ParameterSet>("matching");
  const auto &referenceDataset = c_m.getParameter<std::string>("reference_dataset");

  // constructing vectors with reference data
  if (!referenceDataset.empty()) {
    TFile *f_ref = TFile::Open(referenceDataset.c_str());
    if (!f_ref->IsOpen()) {
      edm::LogWarning("PPS") << "[ESSource] could not find reference dataset file: " << referenceDataset;
    } else {
      TDirectory *ad_ref = findDirectoryWithName((TDirectory *)f_ref, sectorConfig45.name_);
      if (ad_ref == nullptr) {
        edm::LogWarning("PPS") << "[ESSource] could not find reference dataset in " << referenceDataset;
      } else {
        edm::LogInfo("PPS") << "[ESSource] loading reference dataset from " << ad_ref->GetPath();

        for (const auto &p : rpTags) {
          if (debug)
            gDirectory = debugFile->mkdir(rpConfigs[p.first]->name_.c_str())->mkdir("fits_ref");

          auto *d_ref = (TDirectory *)ad_ref->Get(
              (sectorNames[p.first] + "/near_far/x slices, " + rpConfigs[p.first]->position_).c_str());
          if (d_ref == nullptr) {
            edm::LogWarning("PPS") << "[ESSource] could not load d_ref";
          } else {
            matchingReferencePoints[p.first] = buildVectorFromDirectory(d_ref, *rpConfigs[p.first]);
          }
        }
      }
    }
    delete f_ref;
  }

  for (const auto &p : rpTags) {
    const auto &ps = c_m.getParameter<edm::ParameterSet>(p.second);
    matchingShiftRanges[p.first] = {ps.getParameter<double>("sh_min"), ps.getParameter<double>("sh_max")};
  }

  const auto &c_axr = iConfig.getParameter<edm::ParameterSet>("x_alignment_relative");
  for (const auto &p : rpTags) {
    const auto &ps = c_axr.getParameter<edm::ParameterSet>(p.second);
    alignment_x_relative_ranges[p.first] = {ps.getParameter<double>("x_min"), ps.getParameter<double>("x_max")};
  }
  nearFarMinEntries = c_axr.getParameter<unsigned int>("near_far_min_entries");

  const auto &c_ay = iConfig.getParameter<edm::ParameterSet>("y_alignment");
  for (const auto &p : rpTags) {
    const auto &ps = c_ay.getParameter<edm::ParameterSet>(p.second);
    alignment_y_ranges[p.first] = {ps.getParameter<double>("x_min"), ps.getParameter<double>("x_max")};
  }
  modeGraphMinN = c_ay.getParameter<unsigned int>("mode_graph_min_N");
  multSelProjYMinEntries = c_ay.getParameter<unsigned int>("mult_sel_proj_y_min_entries");

  const auto &bps = iConfig.getParameter<edm::ParameterSet>("binning");
  binning.bin_size_x_ = bps.getParameter<double>("bin_size_x");
  binning.n_bins_x_ = bps.getParameter<unsigned int>("n_bins_x");
  binning.pixel_x_offset_ = bps.getParameter<double>("pixel_x_offset");
  binning.n_bins_y_ = bps.getParameter<unsigned int>("n_bins_y");
  binning.y_min_ = bps.getParameter<double>("y_min");
  binning.y_max_ = bps.getParameter<double>("y_max");

  setWhatProduced(this, label);
  findingRecord<PPSAlignmentConfigRcd>();

  if (debug)
    delete debugFile;
}

//---------------------------------------------------------------------------------------------

std::unique_ptr<PPSAlignmentConfig> PPSAlignmentConfigESSource::produce(const PPSAlignmentConfigRcd &) {
  auto p = std::make_unique<PPSAlignmentConfig>();

  p->setSequence(sequence);
  p->setResultsDir(resultsDir);

  p->setSectorConfig45(sectorConfig45);
  p->setSectorConfig56(sectorConfig56);

  p->setX_ali_sh_step(x_ali_sh_step);

  p->setY_mode_sys_unc(y_mode_sys_unc);
  p->setChiSqThreshold(chiSqThreshold);
  p->setY_mode_unc_max_valid(y_mode_unc_max_valid);
  p->setY_mode_max_valid(y_mode_max_valid);

  p->setMaxRPTracksSize(maxRPTracksSize);
  p->setN_si(n_si);

  p->setMatchingReferencePoints(matchingReferencePoints);
  p->setMatchingShiftRanges(matchingShiftRanges);

  p->setAlignment_x_meth_o_ranges(alignment_x_meth_o_ranges);
  p->setFitProfileMinBinEntries(fitProfileMinBinEntries);
  p->setFitProfileMinNReasonable(fitProfileMinNReasonable);
  p->setMethOGraphMinN(methOGraphMinN);
  p->setMethOUncFitRange(methOUncFitRange);

  p->setAlignment_x_relative_ranges(alignment_x_relative_ranges);
  p->setNearFarMinEntries(nearFarMinEntries);

  p->setAlignment_y_ranges(alignment_y_ranges);
  p->setModeGraphMinN(modeGraphMinN);
  p->setMultSelProjYMinEntries(multSelProjYMinEntries);

  p->setBinning(binning);

  edm::LogInfo("PPS") << "\n"
                      << "[ESSource] " << (label.empty() ? "empty label" : "label = " + label) << ":\n\n"
                      << (*p);

  return p;
}

//---------------------------------------------------------------------------------------------

// most default values come from 2018 period
void PPSAlignmentConfigESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("debug", false);

  desc.add<std::string>("label", "");

  desc.add<std::vector<std::string>>("sequence", {});
  desc.add<std::string>("results_dir", "./alignment_results.txt");

  // sector_45
  {
    edm::ParameterSetDescription sector45;

    edm::ParameterSetDescription rp_N;
    rp_N.add<std::string>("name", "L_1_F");
    rp_N.add<int>("id", 3);

    rp_N.add<double>("slope", 0.19);
    rp_N.add<double>("sh_x", -3.6);

    rp_N.add<double>("x_min_fit_mode", 2.);
    rp_N.add<double>("x_max_fit_mode", 7.0);
    rp_N.add<double>("y_max_fit_mode", 7.0);
    rp_N.add<double>("y_cen_add", -0.3);
    rp_N.add<double>("y_width_mult", 1.1);

    rp_N.add<double>("x_slice_min", 7.);
    rp_N.add<double>("x_slice_max", 19.);
    rp_N.add<double>("x_slice_w", 0.2);
    sector45.add<edm::ParameterSetDescription>("rp_N", rp_N);

    edm::ParameterSetDescription rp_F;
    rp_F.add<std::string>("name", "L_2_F");
    rp_F.add<int>("id", 23);

    rp_F.add<double>("slope", 0.19);
    rp_F.add<double>("sh_x", -42.);

    rp_F.add<double>("x_min_fit_mode", 2.);
    rp_F.add<double>("x_max_fit_mode", 7.5);
    rp_F.add<double>("y_max_fit_mode", 7.5);
    rp_F.add<double>("y_cen_add", -0.3);
    rp_F.add<double>("y_width_mult", 1.1);

    rp_F.add<double>("x_slice_min", 46.);
    rp_F.add<double>("x_slice_max", 58.);
    rp_F.add<double>("x_slice_w", 0.2);
    sector45.add<edm::ParameterSetDescription>("rp_F", rp_F);

    sector45.add<double>("slope", 0.006);
    sector45.add<bool>("cut_h_apply", true);
    sector45.add<double>("cut_h_a", -1.);
    sector45.add<double>("cut_h_c", -38.55);
    sector45.add<double>("cut_h_si", 0.2);
    sector45.add<bool>("cut_v_apply", true);
    sector45.add<double>("cut_v_a", -1.07);
    sector45.add<double>("cut_v_c", 1.63);
    sector45.add<double>("cut_v_si", 0.15);

    desc.add<edm::ParameterSetDescription>("sector_45", sector45);
  }

  // sector_56
  {
    edm::ParameterSetDescription sector56;

    edm::ParameterSetDescription rp_N;
    rp_N.add<std::string>("name", "R_1_F");
    rp_N.add<int>("id", 103);

    rp_N.add<double>("slope", 0.40);
    rp_N.add<double>("sh_x", -2.8);

    rp_N.add<double>("x_min_fit_mode", 2.);
    rp_N.add<double>("x_max_fit_mode", 7.4);
    rp_N.add<double>("y_max_fit_mode", 7.4);
    rp_N.add<double>("y_cen_add", -0.8);
    rp_N.add<double>("y_width_mult", 1.0);

    rp_N.add<double>("x_slice_min", 6.);
    rp_N.add<double>("x_slice_max", 17.);
    rp_N.add<double>("x_slice_w", 0.2);
    sector56.add<edm::ParameterSetDescription>("rp_N", rp_N);

    edm::ParameterSetDescription rp_F;
    rp_F.add<std::string>("name", "R_2_F");
    rp_F.add<int>("id", 123);

    rp_F.add<double>("slope", 0.39);
    rp_F.add<double>("sh_x", -41.9);

    rp_F.add<double>("x_min_fit_mode", 2.);
    rp_F.add<double>("x_max_fit_mode", 8.0);
    rp_F.add<double>("y_max_fit_mode", 8.0);
    rp_F.add<double>("y_cen_add", -0.8);
    rp_F.add<double>("y_width_mult", 1.0);

    rp_F.add<double>("x_slice_min", 45.);
    rp_F.add<double>("x_slice_max", 57.);
    rp_F.add<double>("x_slice_w", 0.2);
    sector56.add<edm::ParameterSetDescription>("rp_F", rp_F);

    sector56.add<double>("slope", -0.015);
    sector56.add<bool>("cut_h_apply", true);
    sector56.add<double>("cut_h_a", -1.);
    sector56.add<double>("cut_h_c", -39.26);
    sector56.add<double>("cut_h_si", 0.2);
    sector56.add<bool>("cut_v_apply", true);
    sector56.add<double>("cut_v_a", -1.07);
    sector56.add<double>("cut_v_c", 1.49);
    sector56.add<double>("cut_v_si", 0.15);

    desc.add<edm::ParameterSetDescription>("sector_56", sector56);
  }

  desc.add<double>("x_ali_sh_step", 0.01);

  desc.add<double>("y_mode_sys_unc", 0.03);
  desc.add<double>("chiSqThreshold", 50.);
  desc.add<double>("y_mode_unc_max_valid", 5.);
  desc.add<double>("y_mode_max_valid", 20.);

  desc.add<unsigned int>("max_RP_tracks_size", 2);
  desc.add<double>("n_si", 4.);

  // matching
  {
    edm::ParameterSetDescription matching;
    matching.add<std::string>("reference_dataset", "");

    edm::ParameterSetDescription rpLF;
    rpLF.add<double>("sh_min", -43.);
    rpLF.add<double>("sh_max", -41.);
    matching.add<edm::ParameterSetDescription>("rp_L_F", rpLF);

    edm::ParameterSetDescription rpLN;
    rpLN.add<double>("sh_min", -4.2);
    rpLN.add<double>("sh_max", -2.4);
    matching.add<edm::ParameterSetDescription>("rp_L_N", rpLN);

    edm::ParameterSetDescription rpRN;
    rpRN.add<double>("sh_min", -3.6);
    rpRN.add<double>("sh_max", -1.8);
    matching.add<edm::ParameterSetDescription>("rp_R_N", rpRN);

    edm::ParameterSetDescription rpRF;
    rpRF.add<double>("sh_min", -43.2);
    rpRF.add<double>("sh_max", -41.2);
    matching.add<edm::ParameterSetDescription>("rp_R_F", rpRF);

    desc.add<edm::ParameterSetDescription>("matching", matching);
  }

  // x alignment meth o
  {
    edm::ParameterSetDescription x_alignment_meth_o;

    edm::ParameterSetDescription rpLF;
    rpLF.add<double>("x_min", 47.);
    rpLF.add<double>("x_max", 56.5);
    x_alignment_meth_o.add<edm::ParameterSetDescription>("rp_L_F", rpLF);

    edm::ParameterSetDescription rpLN;
    rpLN.add<double>("x_min", 9.);
    rpLN.add<double>("x_max", 18.5);
    x_alignment_meth_o.add<edm::ParameterSetDescription>("rp_L_N", rpLN);

    edm::ParameterSetDescription rpRN;
    rpRN.add<double>("x_min", 7.);
    rpRN.add<double>("x_max", 15.);
    x_alignment_meth_o.add<edm::ParameterSetDescription>("rp_R_N", rpRN);

    edm::ParameterSetDescription rpRF;
    rpRF.add<double>("x_min", 46.);
    rpRF.add<double>("x_max", 54.);
    x_alignment_meth_o.add<edm::ParameterSetDescription>("rp_R_F", rpRF);

    x_alignment_meth_o.add<unsigned int>("fit_profile_min_bin_entries", 5);
    x_alignment_meth_o.add<unsigned int>("fit_profile_min_N_reasonable", 10);
    x_alignment_meth_o.add<unsigned int>("meth_o_graph_min_N", 5);
    x_alignment_meth_o.add<double>("meth_o_unc_fit_range", 0.5);

    desc.add<edm::ParameterSetDescription>("x_alignment_meth_o", x_alignment_meth_o);
  }

  // x alignment relative
  {
    edm::ParameterSetDescription x_alignment_relative;

    edm::ParameterSetDescription rpLF;
    rpLF.add<double>("x_min", 0.);
    rpLF.add<double>("x_max", 0.);
    x_alignment_relative.add<edm::ParameterSetDescription>("rp_L_F", rpLF);

    edm::ParameterSetDescription rpLN;
    rpLN.add<double>("x_min", 7.5);
    rpLN.add<double>("x_max", 12.);
    x_alignment_relative.add<edm::ParameterSetDescription>("rp_L_N", rpLN);

    edm::ParameterSetDescription rpRN;
    rpRN.add<double>("x_min", 6.);
    rpRN.add<double>("x_max", 10.);
    x_alignment_relative.add<edm::ParameterSetDescription>("rp_R_N", rpRN);

    edm::ParameterSetDescription rpRF;
    rpRF.add<double>("x_min", 0.);
    rpRF.add<double>("x_max", 0.);
    x_alignment_relative.add<edm::ParameterSetDescription>("rp_R_F", rpRF);

    x_alignment_relative.add<unsigned int>("near_far_min_entries", 100);

    desc.add<edm::ParameterSetDescription>("x_alignment_relative", x_alignment_relative);
  }

  // y alignment
  {
    edm::ParameterSetDescription y_alignment;

    edm::ParameterSetDescription rpLF;
    rpLF.add<double>("x_min", 44.5);
    rpLF.add<double>("x_max", 49.);
    y_alignment.add<edm::ParameterSetDescription>("rp_L_F", rpLF);

    edm::ParameterSetDescription rpLN;
    rpLN.add<double>("x_min", 6.7);
    rpLN.add<double>("x_max", 11.);
    y_alignment.add<edm::ParameterSetDescription>("rp_L_N", rpLN);

    edm::ParameterSetDescription rpRN;
    rpRN.add<double>("x_min", 5.9);
    rpRN.add<double>("x_max", 10.);
    y_alignment.add<edm::ParameterSetDescription>("rp_R_N", rpRN);

    edm::ParameterSetDescription rpRF;
    rpRF.add<double>("x_min", 44.5);
    rpRF.add<double>("x_max", 49.);
    y_alignment.add<edm::ParameterSetDescription>("rp_R_F", rpRF);

    y_alignment.add<unsigned int>("mode_graph_min_N", 5);
    y_alignment.add<unsigned int>("mult_sel_proj_y_min_entries", 300);

    desc.add<edm::ParameterSetDescription>("y_alignment", y_alignment);
  }

  // binning
  {
    edm::ParameterSetDescription binning;

    binning.add<double>("bin_size_x", 142.3314E-3);
    binning.add<unsigned int>("n_bins_x", 210);
    binning.add<double>("pixel_x_offset", 40.);
    binning.add<unsigned int>("n_bins_y", 400);
    binning.add<double>("y_min", -20.);
    binning.add<double>("y_max", 20.);

    desc.add<edm::ParameterSetDescription>("binning", binning);
  }

  descriptions.add("ppsAlignmentConfigESSource", desc);
}

//---------------------------------------------------------------------------------------------

// Fits a linear function to a TProfile (similar method in PPSAlignmentHarvester).
int PPSAlignmentConfigESSource::fitProfile(TProfile *p, double x_mean, double x_rms, double &sl, double &sl_unc) {
  unsigned int n_reasonable = 0;
  for (int bi = 1; bi <= p->GetNbinsX(); bi++) {
    if (p->GetBinEntries(bi) < fitProfileMinBinEntries) {
      p->SetBinContent(bi, 0.);
      p->SetBinError(bi, 0.);
    } else {
      n_reasonable++;
    }
  }

  if (n_reasonable < fitProfileMinNReasonable)
    return 1;

  double xMin = x_mean - x_rms, xMax = x_mean + x_rms;

  TF1 *ff_pol1 = new TF1("ff_pol1", "[0] + [1]*x");

  ff_pol1->SetParameter(0., 0.);
  p->Fit(ff_pol1, "Q", "", xMin, xMax);

  sl = ff_pol1->GetParameter(1);
  sl_unc = ff_pol1->GetParError(1);

  return 0;
}

//---------------------------------------------------------------------------------------------

// Performs a breadth first search on dir. If found, returns the directory with object
// named searchName inside. Otherwise, returns nullptr.
TDirectory *PPSAlignmentConfigESSource::findDirectoryWithName(TDirectory *dir, std::string searchName) {
  TIter next(dir->GetListOfKeys());
  std::queue<TDirectory *> dirQueue;
  TObject *o;
  while ((o = next())) {
    TKey *k = (TKey *)o;

    std::string name = k->GetName();
    if (name == searchName)
      return dir;

    if (((TSystemFile *)k)->IsDirectory())
      dirQueue.push((TDirectory *)k->ReadObj());
  }

  while (!dirQueue.empty()) {
    TDirectory *resultDir = findDirectoryWithName(dirQueue.front(), searchName);
    dirQueue.pop();
    if (resultDir != nullptr)
      return resultDir;
  }

  return nullptr;
}

//---------------------------------------------------------------------------------------------

// Builds vector of PointErrors instances from slice plots in dir.
std::vector<PointErrors> PPSAlignmentConfigESSource::buildVectorFromDirectory(TDirectory *dir, const RPConfig &rpd) {
  std::vector<PointErrors> pv;

  TIter next(dir->GetListOfKeys());
  TObject *o;
  while ((o = next())) {
    TKey *k = (TKey *)o;

    std::string name = k->GetName();
    size_t d = name.find('-');
    const double x_min = std::stod(name.substr(0, d));
    const double x_max = std::stod(name.substr(d + 1));

    TDirectory *d_slice = (TDirectory *)k->ReadObj();

    TH1D *h_y = (TH1D *)d_slice->Get("h_y");
    TProfile *p_y_diffFN_vs_y = (TProfile *)d_slice->Get("p_y_diffFN_vs_y");

    double y_cen = h_y->GetMean();
    double y_width = h_y->GetRMS();

    y_cen += rpd.y_cen_add_;
    y_width *= rpd.y_width_mult_;

    double sl = 0., sl_unc = 0.;
    int fr = fitProfile(p_y_diffFN_vs_y, y_cen, y_width, sl, sl_unc);
    if (fr != 0)
      continue;

    if (debug)
      p_y_diffFN_vs_y->Write(name.c_str());

    pv.push_back({(x_max + x_min) / 2., sl, (x_max - x_min) / 2., sl_unc});
  }

  return pv;
}

//---------------------------------------------------------------------------------------------

void PPSAlignmentConfigESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                const edm::IOVSyncValue &iosv,
                                                edm::ValidityInterval &oValidity) {
  edm::LogInfo("PPS") << ">> PPSAlignmentConfigESSource::setIntervalFor(" << key.name() << ")\n"
                      << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_SOURCE(PPSAlignmentConfigESSource);