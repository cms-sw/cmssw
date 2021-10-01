#ifndef CondFormats_CSCObjects_CSCL1TPLookupTableME21ILT_h
#define CondFormats_CSCObjects_CSCL1TPLookupTableME21ILT_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class CSCL1TPLookupTableME21ILT {
public:
  CSCL1TPLookupTableME21ILT();

  ~CSCL1TPLookupTableME21ILT() {}

  typedef std::vector<unsigned> t_lut;

  // setters
  void set_GEM_pad_CSC_hs_ME21_even(t_lut lut);
  void set_GEM_pad_CSC_hs_ME21_odd(t_lut lut);

  void set_GEM_pad_CSC_es_ME21_even(t_lut lut);
  void set_GEM_pad_CSC_es_ME21_odd(t_lut lut);

  void set_GEM_roll_L1_CSC_min_wg_ME21_even(t_lut lut);
  void set_GEM_roll_L1_CSC_max_wg_ME21_even(t_lut lut);
  void set_GEM_roll_L1_CSC_min_wg_ME21_odd(t_lut lut);
  void set_GEM_roll_L1_CSC_max_wg_ME21_odd(t_lut lut);

  void set_GEM_roll_L2_CSC_min_wg_ME21_even(t_lut lut);
  void set_GEM_roll_L2_CSC_max_wg_ME21_even(t_lut lut);
  void set_GEM_roll_L2_CSC_min_wg_ME21_odd(t_lut lut);
  void set_GEM_roll_L2_CSC_max_wg_ME21_odd(t_lut lut);

  void set_es_diff_slope_L1_ME21_even(t_lut lut);
  void set_es_diff_slope_L2_ME21_even(t_lut lut);
  void set_es_diff_slope_L1_ME21_odd(t_lut lut);
  void set_es_diff_slope_L2_ME21_odd(t_lut lut);

  // getters
  unsigned GEM_pad_CSC_hs_ME21_even(unsigned pad) const;
  unsigned GEM_pad_CSC_hs_ME21_odd(unsigned pad) const;

  unsigned GEM_pad_CSC_es_ME21_even(unsigned pad) const;
  unsigned GEM_pad_CSC_es_ME21_odd(unsigned pad) const;

  unsigned GEM_roll_L1_CSC_min_wg_ME21_even(unsigned roll) const;
  unsigned GEM_roll_L1_CSC_max_wg_ME21_even(unsigned roll) const;
  unsigned GEM_roll_L1_CSC_min_wg_ME21_odd(unsigned roll) const;
  unsigned GEM_roll_L1_CSC_max_wg_ME21_odd(unsigned roll) const;

  unsigned GEM_roll_L2_CSC_min_wg_ME21_even(unsigned roll) const;
  unsigned GEM_roll_L2_CSC_max_wg_ME21_even(unsigned roll) const;
  unsigned GEM_roll_L2_CSC_min_wg_ME21_odd(unsigned roll) const;
  unsigned GEM_roll_L2_CSC_max_wg_ME21_odd(unsigned roll) const;

  // GEM-CSC trigger: 1/8-strip difference to slope
  unsigned es_diff_slope_L1_ME21_even(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME21_even(unsigned es_diff) const;
  unsigned es_diff_slope_L1_ME21_odd(unsigned es_diff) const;
  unsigned es_diff_slope_L2_ME21_odd(unsigned es_diff) const;

private:
  std::vector<unsigned> GEM_pad_CSC_hs_ME21_even_;
  std::vector<unsigned> GEM_pad_CSC_hs_ME21_odd_;

  std::vector<unsigned> GEM_pad_CSC_es_ME21_even_;
  std::vector<unsigned> GEM_pad_CSC_es_ME21_odd_;

  std::vector<unsigned> GEM_roll_L1_CSC_min_wg_ME21_even_;
  std::vector<unsigned> GEM_roll_L1_CSC_max_wg_ME21_even_;
  std::vector<unsigned> GEM_roll_L1_CSC_min_wg_ME21_odd_;
  std::vector<unsigned> GEM_roll_L1_CSC_max_wg_ME21_odd_;

  std::vector<unsigned> GEM_roll_L2_CSC_min_wg_ME21_even_;
  std::vector<unsigned> GEM_roll_L2_CSC_max_wg_ME21_even_;
  std::vector<unsigned> GEM_roll_L2_CSC_min_wg_ME21_odd_;
  std::vector<unsigned> GEM_roll_L2_CSC_max_wg_ME21_odd_;

  std::vector<unsigned> es_diff_slope_L1_ME21_even_;
  std::vector<unsigned> es_diff_slope_L2_ME21_even_;
  std::vector<unsigned> es_diff_slope_L1_ME21_odd_;
  std::vector<unsigned> es_diff_slope_L2_ME21_odd_;

  COND_SERIALIZABLE;
};

#endif
