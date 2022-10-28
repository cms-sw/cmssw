#ifndef CondFormats_CSCObjects_CSCL1TPLookupTableME11ILT_h
#define CondFormats_CSCObjects_CSCL1TPLookupTableME11ILT_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class CSCL1TPLookupTableME11ILT {
public:
  CSCL1TPLookupTableME11ILT();

  ~CSCL1TPLookupTableME11ILT() {}

  typedef std::vector<unsigned> t_lut;

  // setters
  void set_GEM_pad_CSC_es_ME11b_even(t_lut lut);
  void set_GEM_pad_CSC_es_ME11a_even(t_lut lut);
  void set_GEM_pad_CSC_es_ME11b_odd(t_lut lut);
  void set_GEM_pad_CSC_es_ME11a_odd(t_lut lut);

  void set_GEM_roll_CSC_min_wg_ME11_even(t_lut lut);
  void set_GEM_roll_CSC_max_wg_ME11_even(t_lut lut);
  void set_GEM_roll_CSC_min_wg_ME11_odd(t_lut lut);
  void set_GEM_roll_CSC_max_wg_ME11_odd(t_lut lut);

  // GEM-CSC trigger: slope correction
  void set_CSC_slope_cosi_2to1_L1_ME11a_even(t_lut lut);
  void set_CSC_slope_cosi_2to1_L1_ME11a_odd(t_lut lut);
  void set_CSC_slope_cosi_3to1_L1_ME11a_even(t_lut lut);
  void set_CSC_slope_cosi_3to1_L1_ME11a_odd(t_lut lut);

  void set_CSC_slope_cosi_2to1_L1_ME11b_even(t_lut lut);
  void set_CSC_slope_cosi_2to1_L1_ME11b_odd(t_lut lut);
  void set_CSC_slope_cosi_3to1_L1_ME11b_even(t_lut lut);
  void set_CSC_slope_cosi_3to1_L1_ME11b_odd(t_lut lut);

  void set_CSC_slope_cosi_corr_L1_ME11a_even(t_lut lut);
  void set_CSC_slope_cosi_corr_L1_ME11b_even(t_lut lut);
  void set_CSC_slope_cosi_corr_L1_ME11a_odd(t_lut lut);
  void set_CSC_slope_cosi_corr_L1_ME11b_odd(t_lut lut);

  void set_CSC_slope_corr_L1_ME11a_even(t_lut lut);
  void set_CSC_slope_corr_L1_ME11b_even(t_lut lut);
  void set_CSC_slope_corr_L1_ME11a_odd(t_lut lut);
  void set_CSC_slope_corr_L1_ME11b_odd(t_lut lut);
  void set_CSC_slope_corr_L2_ME11a_even(t_lut lut);
  void set_CSC_slope_corr_L2_ME11b_even(t_lut lut);
  void set_CSC_slope_corr_L2_ME11a_odd(t_lut lut);
  void set_CSC_slope_corr_L2_ME11b_odd(t_lut lut);

  void set_es_diff_slope_L1_ME11a_even(t_lut lut);
  void set_es_diff_slope_L1_ME11a_odd(t_lut lut);
  void set_es_diff_slope_L1_ME11b_even(t_lut lut);
  void set_es_diff_slope_L1_ME11b_odd(t_lut lut);
  void set_es_diff_slope_L2_ME11a_even(t_lut lut);
  void set_es_diff_slope_L2_ME11a_odd(t_lut lut);
  void set_es_diff_slope_L2_ME11b_even(t_lut lut);
  void set_es_diff_slope_L2_ME11b_odd(t_lut lut);

  // getters
  unsigned GEM_pad_CSC_es_ME11b_even(unsigned pad) const;
  unsigned GEM_pad_CSC_es_ME11a_even(unsigned pad) const;
  unsigned GEM_pad_CSC_es_ME11b_odd(unsigned pad) const;
  unsigned GEM_pad_CSC_es_ME11a_odd(unsigned pad) const;

  unsigned GEM_roll_CSC_min_wg_ME11_even(unsigned roll) const;
  unsigned GEM_roll_CSC_max_wg_ME11_even(unsigned roll) const;
  unsigned GEM_roll_CSC_min_wg_ME11_odd(unsigned roll) const;
  unsigned GEM_roll_CSC_max_wg_ME11_odd(unsigned roll) const;

  // GEM-CSC trigger: slope correction
  unsigned CSC_slope_cosi_2to1_L1_ME11a_even(unsigned channel) const;
  unsigned CSC_slope_cosi_2to1_L1_ME11a_odd(unsigned channel) const;
  unsigned CSC_slope_cosi_3to1_L1_ME11a_even(unsigned channel) const;
  unsigned CSC_slope_cosi_3to1_L1_ME11a_odd(unsigned channel) const;

  unsigned CSC_slope_cosi_2to1_L1_ME11b_even(unsigned channel) const;
  unsigned CSC_slope_cosi_2to1_L1_ME11b_odd(unsigned channel) const;
  unsigned CSC_slope_cosi_3to1_L1_ME11b_even(unsigned channel) const;
  unsigned CSC_slope_cosi_3to1_L1_ME11b_odd(unsigned channel) const;

  unsigned CSC_slope_cosi_corr_L1_ME11a_even(unsigned channel) const;
  unsigned CSC_slope_cosi_corr_L1_ME11b_even(unsigned channel) const;
  unsigned CSC_slope_cosi_corr_L1_ME11a_odd(unsigned channel) const;
  unsigned CSC_slope_cosi_corr_L1_ME11b_odd(unsigned channel) const;

  unsigned CSC_slope_corr_L1_ME11a_even(unsigned channel) const;
  unsigned CSC_slope_corr_L1_ME11b_even(unsigned channel) const;
  unsigned CSC_slope_corr_L1_ME11a_odd(unsigned channel) const;
  unsigned CSC_slope_corr_L1_ME11b_odd(unsigned channel) const;
  unsigned CSC_slope_corr_L2_ME11a_even(unsigned channel) const;
  unsigned CSC_slope_corr_L2_ME11b_even(unsigned channel) const;
  unsigned CSC_slope_corr_L2_ME11a_odd(unsigned channel) const;
  unsigned CSC_slope_corr_L2_ME11b_odd(unsigned channel) const;

  // GEM-CSC trigger: 1/8-strip difference to slope
  unsigned es_diff_slope_L1_ME11a_even(unsigned es_diff) const;
  unsigned es_diff_slope_L1_ME11a_odd(unsigned es_diff) const;
  unsigned es_diff_slope_L1_ME11b_even(unsigned es_diff) const;
  unsigned es_diff_slope_L1_ME11b_odd(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME11a_even(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME11a_odd(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME11b_even(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME11b_odd(unsigned es_diff) const;

private:
  t_lut GEM_pad_CSC_es_ME11b_even_;
  t_lut GEM_pad_CSC_es_ME11a_even_;
  t_lut GEM_pad_CSC_es_ME11b_odd_;
  t_lut GEM_pad_CSC_es_ME11a_odd_;

  t_lut GEM_roll_CSC_min_wg_ME11_even_;
  t_lut GEM_roll_CSC_max_wg_ME11_even_;
  t_lut GEM_roll_CSC_min_wg_ME11_odd_;
  t_lut GEM_roll_CSC_max_wg_ME11_odd_;

  t_lut CSC_slope_cosi_2to1_L1_ME11a_even_;
  t_lut CSC_slope_cosi_2to1_L1_ME11a_odd_;
  t_lut CSC_slope_cosi_3to1_L1_ME11a_even_;
  t_lut CSC_slope_cosi_3to1_L1_ME11a_odd_;

  t_lut CSC_slope_cosi_2to1_L1_ME11b_even_;
  t_lut CSC_slope_cosi_2to1_L1_ME11b_odd_;
  t_lut CSC_slope_cosi_3to1_L1_ME11b_even_;
  t_lut CSC_slope_cosi_3to1_L1_ME11b_odd_;

  t_lut CSC_slope_cosi_corr_L1_ME11a_even_;
  t_lut CSC_slope_cosi_corr_L1_ME11b_even_;
  t_lut CSC_slope_cosi_corr_L1_ME11a_odd_;
  t_lut CSC_slope_cosi_corr_L1_ME11b_odd_;

  t_lut CSC_slope_corr_L1_ME11a_even_;
  t_lut CSC_slope_corr_L1_ME11b_even_;
  t_lut CSC_slope_corr_L1_ME11a_odd_;
  t_lut CSC_slope_corr_L1_ME11b_odd_;
  t_lut CSC_slope_corr_L2_ME11a_even_;
  t_lut CSC_slope_corr_L2_ME11b_even_;
  t_lut CSC_slope_corr_L2_ME11a_odd_;
  t_lut CSC_slope_corr_L2_ME11b_odd_;

  t_lut es_diff_slope_L1_ME11a_even_;
  t_lut es_diff_slope_L1_ME11a_odd_;
  t_lut es_diff_slope_L1_ME11b_even_;
  t_lut es_diff_slope_L1_ME11b_odd_;
  t_lut es_diff_slope_L2_ME11a_even_;
  t_lut es_diff_slope_L2_ME11a_odd_;
  t_lut es_diff_slope_L2_ME11b_even_;
  t_lut es_diff_slope_L2_ME11b_odd_;

  COND_SERIALIZABLE;
};

#endif
