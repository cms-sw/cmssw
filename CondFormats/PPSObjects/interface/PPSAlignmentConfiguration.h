/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#ifndef CondFormats_PPSObjects_PPSAlignmentConfiguration_h
#define CondFormats_PPSObjects_PPSAlignmentConfiguration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <map>

class PPSAlignmentConfiguration {
public:
  // Auxiliary structures
  struct PointErrors {
    double x_;
    double y_;
    double ex_;  // error x
    double ey_;  // error y

    COND_SERIALIZABLE;
  };

  struct SelectionRange {
    double x_min_;
    double x_max_;

    COND_SERIALIZABLE;
  };

  struct RPConfig {
    std::string name_;
    unsigned int id_;
    std::string position_;
    double slope_;
    double sh_x_;

    double x_min_fit_mode_, x_max_fit_mode_;
    double y_max_fit_mode_;
    double y_cen_add_;
    double y_width_mult_;

    int x_slice_n_;
    double x_slice_min_, x_slice_w_;

    COND_SERIALIZABLE;
  };

  struct SectorConfig {
    std::string name_;
    RPConfig rp_N_, rp_F_;
    double slope_;

    bool cut_h_apply_;
    double cut_h_a_, cut_h_c_, cut_h_si_;

    bool cut_v_apply_;
    double cut_v_a_, cut_v_c_, cut_v_si_;

    COND_SERIALIZABLE;
  };

  struct Binning {
    double bin_size_x_;  // mm
    unsigned int n_bins_x_;

    double pixel_x_offset_;

    unsigned int n_bins_y_;
    double y_min_, y_max_;

    unsigned int diffFN_n_bins_x_;
    double diffFN_x_min_, diffFN_x_max_;

    unsigned int slice_n_bins_x_, slice_n_bins_y_;
    double slice_x_min_, slice_x_max_, slice_y_min_, slice_y_max_;

    COND_SERIALIZABLE;
  };

  // Getters
  const SectorConfig& sectorConfig45() const;
  const SectorConfig& sectorConfig56() const;

  double x_ali_sh_step() const;

  double y_mode_sys_unc() const;
  double chiSqThreshold() const;
  double y_mode_unc_max_valid() const;
  double y_mode_max_valid() const;

  double minRPTracksSize() const;
  double maxRPTracksSize() const;
  double n_si() const;

  const std::map<unsigned int, std::vector<PointErrors>>& matchingReferencePoints() const;
  const std::map<unsigned int, SelectionRange>& matchingShiftRanges() const;

  const std::map<unsigned int, SelectionRange>& alignment_x_meth_o_ranges() const;
  unsigned int fitProfileMinBinEntries() const;
  unsigned int fitProfileMinNReasonable() const;
  unsigned int methOGraphMinN() const;
  double methOUncFitRange() const;

  const std::map<unsigned int, SelectionRange>& alignment_x_relative_ranges() const;
  unsigned int nearFarMinEntries() const;

  const std::map<unsigned int, SelectionRange>& alignment_y_ranges() const;
  unsigned int modeGraphMinN() const;
  unsigned int multSelProjYMinEntries() const;

  const Binning& binning() const;

  const std::vector<double>& extraParams() const;

  // Setters
  void setSectorConfig45(SectorConfig& sectorConfig45);
  void setSectorConfig56(SectorConfig& sectorConfig56);

  void setX_ali_sh_step(double x_ali_sh_step);

  void setY_mode_sys_unc(double y_mode_sys_unc);
  void setChiSqThreshold(double chiSqThreshold);
  void setY_mode_unc_max_valid(double y_mode_unc_max_valid);
  void setY_mode_max_valid(double y_mode_max_valid);

  void setMinRPTracksSize(unsigned int minRPTracksSize);
  void setMaxRPTracksSize(unsigned int maxRPTracksSize);
  void setN_si(double n_si);

  void setMatchingReferencePoints(std::map<unsigned int, std::vector<PointErrors>>& matchingReferencePoints);
  void setMatchingShiftRanges(std::map<unsigned int, SelectionRange>& matchingShiftRanges);

  void setAlignment_x_meth_o_ranges(std::map<unsigned int, SelectionRange>& alignment_x_meth_o_ranges);
  void setFitProfileMinBinEntries(unsigned int fitProfileMinBinEntries);
  void setFitProfileMinNReasonable(unsigned int fitProfileMinNReasonable);
  void setMethOGraphMinN(unsigned int methOGraphMinN);
  void setMethOUncFitRange(double methOUncFitRange);

  void setAlignment_x_relative_ranges(std::map<unsigned int, SelectionRange>& alignment_x_relative_ranges);
  void setNearFarMinEntries(unsigned int nearFarMinEntries);

  void setAlignment_y_ranges(std::map<unsigned int, SelectionRange>& alignment_y_ranges);
  void setModeGraphMinN(unsigned int modeGraphMinN);
  void setMultSelProjYMinEntries(unsigned int multSelProjYMinEntries);

  void setBinning(Binning& binning);

  void setExtraParams(std::vector<double>& extraParams);

  // << operator
  friend std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration& c);

private:
  SectorConfig sectorConfig45_, sectorConfig56_;

  double x_ali_sh_step_;  // mm

  double y_mode_sys_unc_;
  double chiSqThreshold_;
  double y_mode_unc_max_valid_;
  double y_mode_max_valid_;

  unsigned int minRPTracksSize_;
  unsigned int maxRPTracksSize_;
  double n_si_;

  std::map<unsigned int, std::vector<PointErrors>> matchingReferencePoints_;
  std::map<unsigned int, SelectionRange> matchingShiftRanges_;

  std::map<unsigned int, SelectionRange> alignment_x_meth_o_ranges_;
  unsigned int fitProfileMinBinEntries_;
  unsigned int fitProfileMinNReasonable_;
  unsigned int methOGraphMinN_;
  double methOUncFitRange_;  // mm

  std::map<unsigned int, SelectionRange> alignment_x_relative_ranges_;
  unsigned int nearFarMinEntries_;

  std::map<unsigned int, SelectionRange> alignment_y_ranges_;
  unsigned int modeGraphMinN_;
  unsigned int multSelProjYMinEntries_;

  Binning binning_;

  std::vector<double> extraParams_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::RPConfig& rc);
std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::SectorConfig& sc);
std::ostream& operator<<(std::ostream& os, const PPSAlignmentConfiguration::Binning& b);

#endif
