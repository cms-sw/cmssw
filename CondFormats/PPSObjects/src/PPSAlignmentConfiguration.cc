/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"

#include <iostream>
#include <cmath>
#include <iomanip>

// -------------------------------- PPSAlignmentConfiguration getters --------------------------------

const PPSAlignmentConfiguration::SectorConfig& PPSAlignmentConfiguration::sectorConfig45() const {
  return sectorConfig45_;
}
const PPSAlignmentConfiguration::SectorConfig& PPSAlignmentConfiguration::sectorConfig56() const {
  return sectorConfig56_;
}

double PPSAlignmentConfiguration::x_ali_sh_step() const { return x_ali_sh_step_; }

double PPSAlignmentConfiguration::y_mode_sys_unc() const { return y_mode_sys_unc_; }
double PPSAlignmentConfiguration::chiSqThreshold() const { return chiSqThreshold_; }
double PPSAlignmentConfiguration::y_mode_unc_max_valid() const { return y_mode_unc_max_valid_; }
double PPSAlignmentConfiguration::y_mode_max_valid() const { return y_mode_max_valid_; }

double PPSAlignmentConfiguration::minRPTracksSize() const { return minRPTracksSize_; }
double PPSAlignmentConfiguration::maxRPTracksSize() const { return maxRPTracksSize_; }
double PPSAlignmentConfiguration::n_si() const { return n_si_; }

const std::map<unsigned int, std::vector<PPSAlignmentConfiguration::PointErrors>>&
PPSAlignmentConfiguration::matchingReferencePoints() const {
  return matchingReferencePoints_;
}
const std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>&
PPSAlignmentConfiguration::matchingShiftRanges() const {
  return matchingShiftRanges_;
}

const std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>&
PPSAlignmentConfiguration::alignment_x_meth_o_ranges() const {
  return alignment_x_meth_o_ranges_;
}
unsigned int PPSAlignmentConfiguration::fitProfileMinBinEntries() const { return fitProfileMinBinEntries_; }
unsigned int PPSAlignmentConfiguration::fitProfileMinNReasonable() const { return fitProfileMinNReasonable_; }
unsigned int PPSAlignmentConfiguration::methOGraphMinN() const { return methOGraphMinN_; }
double PPSAlignmentConfiguration::methOUncFitRange() const { return methOUncFitRange_; }

const std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>&
PPSAlignmentConfiguration::alignment_x_relative_ranges() const {
  return alignment_x_relative_ranges_;
}
unsigned int PPSAlignmentConfiguration::nearFarMinEntries() const { return nearFarMinEntries_; }

const std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>& PPSAlignmentConfiguration::alignment_y_ranges()
    const {
  return alignment_y_ranges_;
}
unsigned int PPSAlignmentConfiguration::modeGraphMinN() const { return modeGraphMinN_; }
unsigned int PPSAlignmentConfiguration::multSelProjYMinEntries() const { return multSelProjYMinEntries_; }

const PPSAlignmentConfiguration::Binning& PPSAlignmentConfiguration::binning() const { return binning_; }

const std::vector<double>& PPSAlignmentConfiguration::extraParams() const { return extraParams_; }

// -------------------------------- PPSAlignmentConfiguration setters --------------------------------

void PPSAlignmentConfiguration::setSectorConfig45(PPSAlignmentConfiguration::SectorConfig& sectorConfig45) {
  sectorConfig45_ = sectorConfig45;
}
void PPSAlignmentConfiguration::setSectorConfig56(PPSAlignmentConfiguration::SectorConfig& sectorConfig56) {
  sectorConfig56_ = sectorConfig56;
}

void PPSAlignmentConfiguration::setX_ali_sh_step(double x_ali_sh_step) { x_ali_sh_step_ = x_ali_sh_step; }

void PPSAlignmentConfiguration::setY_mode_sys_unc(double y_mode_sys_unc) { y_mode_sys_unc_ = y_mode_sys_unc; }
void PPSAlignmentConfiguration::setChiSqThreshold(double chiSqThreshold) { chiSqThreshold_ = chiSqThreshold; }
void PPSAlignmentConfiguration::setY_mode_unc_max_valid(double y_mode_unc_max_valid) {
  y_mode_unc_max_valid_ = y_mode_unc_max_valid;
}
void PPSAlignmentConfiguration::setY_mode_max_valid(double y_mode_max_valid) { y_mode_max_valid_ = y_mode_max_valid; }

void PPSAlignmentConfiguration::setMinRPTracksSize(unsigned int minRPTracksSize) { minRPTracksSize_ = minRPTracksSize; }
void PPSAlignmentConfiguration::setMaxRPTracksSize(unsigned int maxRPTracksSize) { maxRPTracksSize_ = maxRPTracksSize; }
void PPSAlignmentConfiguration::setN_si(double n_si) { n_si_ = n_si; }

void PPSAlignmentConfiguration::setMatchingReferencePoints(
    std::map<unsigned int, std::vector<PPSAlignmentConfiguration::PointErrors>>& matchingReferencePoints) {
  matchingReferencePoints_ = matchingReferencePoints;
}
void PPSAlignmentConfiguration::setMatchingShiftRanges(
    std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>& matchingShiftRanges) {
  matchingShiftRanges_ = matchingShiftRanges;
}

void PPSAlignmentConfiguration::setAlignment_x_meth_o_ranges(
    std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>& alignment_x_meth_o_ranges) {
  alignment_x_meth_o_ranges_ = alignment_x_meth_o_ranges;
}
void PPSAlignmentConfiguration::setFitProfileMinBinEntries(unsigned int fitProfileMinBinEntries) {
  fitProfileMinBinEntries_ = fitProfileMinBinEntries;
}
void PPSAlignmentConfiguration::setFitProfileMinNReasonable(unsigned int fitProfileMinNReasonable) {
  fitProfileMinNReasonable_ = fitProfileMinNReasonable;
}
void PPSAlignmentConfiguration::setMethOGraphMinN(unsigned int methOGraphMinN) { methOGraphMinN_ = methOGraphMinN; }
void PPSAlignmentConfiguration::setMethOUncFitRange(double methOUncFitRange) { methOUncFitRange_ = methOUncFitRange; }

void PPSAlignmentConfiguration::setAlignment_x_relative_ranges(
    std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>& alignment_x_relative_ranges) {
  alignment_x_relative_ranges_ = alignment_x_relative_ranges;
}
void PPSAlignmentConfiguration::setNearFarMinEntries(unsigned int nearFarMinEntries) {
  nearFarMinEntries_ = nearFarMinEntries;
}

void PPSAlignmentConfiguration::setAlignment_y_ranges(
    std::map<unsigned int, PPSAlignmentConfiguration::SelectionRange>& alignment_y_ranges) {
  alignment_y_ranges_ = alignment_y_ranges;
}
void PPSAlignmentConfiguration::setModeGraphMinN(unsigned int modeGraphMinN) { modeGraphMinN_ = modeGraphMinN; }
void PPSAlignmentConfiguration::setMultSelProjYMinEntries(unsigned int multSelProjYMinEntries) {
  multSelProjYMinEntries_ = multSelProjYMinEntries;
}

void PPSAlignmentConfiguration::setBinning(PPSAlignmentConfiguration::Binning& binning) { binning_ = binning; }

void PPSAlignmentConfiguration::setExtraParams(std::vector<double>& extraParams) { extraParams_ = extraParams; }

// -------------------------------- << operators --------------------------------

std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::RPConfig& rc) {
  os << std::fixed << std::setprecision(3);
  os << "    " << rc.name_ << ", id = " << rc.id_ << ", position = " << rc.position_ << ":\n";
  os << "        slope = " << rc.slope_ << ", sh_x = " << rc.sh_x_ << "\n";
  os << "        x_min_fit_mode = " << rc.x_min_fit_mode_ << ", x_max_fit_mode = " << rc.x_max_fit_mode_ << "\n";
  os << "        y_max_fit_mode = " << rc.y_max_fit_mode_ << "\n";
  os << "        y_cen_add = " << rc.y_cen_add_ << ", y_width_mult = " << rc.y_width_mult_ << "\n";
  os << std::setprecision(2);
  os << "        x slices: min = " << rc.x_slice_min_ << ", w = " << rc.x_slice_w_ << ", n = " << rc.x_slice_n_;

  return os;
}

std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::SectorConfig& sc) {
  os << std::fixed << std::setprecision(3);
  os << sc.name_ << ":\n";
  os << sc.rp_N_ << "\n" << sc.rp_F_ << "\n";
  os << std::setprecision(3);
  os << "    slope = " << sc.slope_ << "\n";
  os << "    cut_h: apply = " << std::boolalpha << sc.cut_h_apply_ << ", a = " << sc.cut_h_a_ << ", c = " << sc.cut_h_c_
     << ", si = " << sc.cut_h_si_ << "\n";
  os << "    cut_v: apply = " << std::boolalpha << sc.cut_v_apply_ << ", a = " << sc.cut_v_a_ << ", c = " << sc.cut_v_c_
     << ", si = " << sc.cut_v_si_ << "\n";

  return os;
}

std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::Binning& b) {
  os << "    bin_size_x = " << b.bin_size_x_ << ", n_bins_x = " << b.n_bins_x_ << "\n";
  os << "    pixel_x_offset = " << b.pixel_x_offset_ << "\n";
  os << "    n_bins_y = " << b.n_bins_y_ << ", y_min = " << b.y_min_ << ", y_max = " << b.y_max_ << "\n";
  os << "    diff far-near:"
     << "\n";
  os << "        n_bins_x = " << b.diffFN_n_bins_x_ << ", x_min = " << b.diffFN_x_min_
     << ", x_max = " << b.diffFN_x_max_ << "\n";
  os << "    slice plots:"
     << "\n";
  os << "        n_bins_x = " << b.slice_n_bins_x_ << ", x_min = " << b.slice_x_min_ << ", x_max = " << b.slice_x_max_
     << "\n";
  os << "        n_bins_y = " << b.slice_n_bins_y_ << ", y_min = " << b.slice_y_min_ << ", y_max = " << b.slice_y_max_;

  return os;
}

std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration& c) {
  os << "* " << c.sectorConfig45_ << "\n\n";
  os << "* " << c.sectorConfig56_ << "\n\n";

  std::map<unsigned int, std::string> rpTags = {{c.sectorConfig45_.rp_F_.id_, c.sectorConfig45_.rp_F_.name_},
                                                {c.sectorConfig45_.rp_N_.id_, c.sectorConfig45_.rp_N_.name_},
                                                {c.sectorConfig56_.rp_N_.id_, c.sectorConfig56_.rp_N_.name_},
                                                {c.sectorConfig56_.rp_F_.id_, c.sectorConfig56_.rp_F_.name_}};

  os << "* x alignment shift step\n";
  os << "    x_ali_sh_step = " << c.x_ali_sh_step_ << "\n\n";

  os << "* mode graph parameters\n";
  os << "    y_mode_sys_unc = " << c.y_mode_sys_unc_ << "\n";
  os << "    chiSqThreshold = " << c.chiSqThreshold_ << "\n";
  os << "    y_mode_unc_max_valid = " << c.y_mode_unc_max_valid_ << "\n";
  os << "    y_mode_max_valid = " << c.y_mode_max_valid_ << "\n\n";

  os << "* selection\n";
  os << "    min_RP_tracks_size = " << c.minRPTracksSize_ << "\n";
  os << "    max_RP_tracks_size = " << c.maxRPTracksSize_ << "\n\n";

  os << "* cuts\n";
  os << "    n_si = " << c.n_si_ << "\n\n";

  os << "* matching\n" << std::setprecision(3);

  os << "    shift ranges:\n";
  for (const auto& p : c.matchingShiftRanges_)
    os << "        RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min_
       << ", sh_max = " << p.second.x_max_ << "\n";

  os << "    reference points:\n";
  for (const auto& pm : c.matchingReferencePoints_) {
    os << "        " << std::setw(3) << pm.first << ": ";
    for (unsigned int i = 0; i < pm.second.size(); i++) {
      const auto& p = pm.second[i];
      if (i % 5 == 0 && i > 0)
        os << "\n             ";
      os << "(" << std::setw(6) << p.x_ << " +- " << p.ex_ << ", " << std::setw(6) << p.y_ << " +- " << p.ey_ << "), ";
    }
    os << "\n";
  }

  os << "\n"
     << "* alignment_x_meth_o\n";
  for (const auto& p : c.alignment_x_meth_o_ranges_)
    os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min_
       << ", sh_max = " << p.second.x_max_ << "\n";
  os << "    fit_profile_min_bin_entries = " << c.fitProfileMinBinEntries_ << "\n";
  os << "    fit_profile_min_N_reasonable = " << c.fitProfileMinNReasonable_ << "\n";
  os << "    meth_o_graph_min_N = " << c.methOGraphMinN_ << "\n";
  os << "    meth_o_unc_fit_range = " << c.methOUncFitRange_ << "\n";

  os << "\n"
     << "* alignment_x_relative\n";
  for (const auto& p : c.alignment_x_relative_ranges_)
    if (p.first == c.sectorConfig45_.rp_N_.id_ || p.first == c.sectorConfig56_.rp_N_.id_) {  // only near RPs
      os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min_
         << ", sh_max = " << p.second.x_max_ << "\n";
    }
  os << "    near_far_min_entries = " << c.nearFarMinEntries_ << "\n";

  os << "\n"
     << "* alignment_y\n";
  for (const auto& p : c.alignment_y_ranges_)
    os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min_
       << ", sh_max = " << p.second.x_max_ << "\n";
  os << "    mode_graph_min_N = " << c.modeGraphMinN_ << "\n";
  os << "    mult_sel_proj_y_min_entries = " << c.multSelProjYMinEntries_ << "\n";

  os << "\n"
     << "* binning\n";
  os << c.binning_ << "\n";

  if (!c.extraParams_.empty()) {
    os << "\n";
    os << "extra_params:\n";
    for (size_t i = 0; i < c.extraParams_.size(); i++) {
      os << std::setw(5) << i << ": " << c.extraParams_[i] << "\n";
    }
  }

  return os;
}
