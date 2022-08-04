#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"

CSCL1TPLookupTableME11ILT::CSCL1TPLookupTableME11ILT()
    : GEM_pad_CSC_es_ME11b_even_(0),
      GEM_pad_CSC_es_ME11a_even_(0),
      GEM_pad_CSC_es_ME11b_odd_(0),
      GEM_pad_CSC_es_ME11a_odd_(0),

      GEM_roll_CSC_min_wg_ME11_even_(0),
      GEM_roll_CSC_max_wg_ME11_even_(0),
      GEM_roll_CSC_min_wg_ME11_odd_(0),
      GEM_roll_CSC_max_wg_ME11_odd_(0),

      CSC_slope_cosi_2to1_L1_ME11a_even_(0),
      CSC_slope_cosi_2to1_L1_ME11a_odd_(0),
      CSC_slope_cosi_3to1_L1_ME11a_even_(0),
      CSC_slope_cosi_3to1_L1_ME11a_odd_(0),

      CSC_slope_cosi_2to1_L1_ME11b_even_(0),
      CSC_slope_cosi_2to1_L1_ME11b_odd_(0),
      CSC_slope_cosi_3to1_L1_ME11b_even_(0),
      CSC_slope_cosi_3to1_L1_ME11b_odd_(0),

      CSC_slope_cosi_corr_L1_ME11a_even_(0),
      CSC_slope_cosi_corr_L1_ME11b_even_(0),
      CSC_slope_cosi_corr_L1_ME11a_odd_(0),
      CSC_slope_cosi_corr_L1_ME11b_odd_(0),

      CSC_slope_corr_L1_ME11a_even_(0),
      CSC_slope_corr_L1_ME11b_even_(0),
      CSC_slope_corr_L1_ME11a_odd_(0),
      CSC_slope_corr_L1_ME11b_odd_(0),
      CSC_slope_corr_L2_ME11a_even_(0),
      CSC_slope_corr_L2_ME11b_even_(0),
      CSC_slope_corr_L2_ME11a_odd_(0),
      CSC_slope_corr_L2_ME11b_odd_(0),

      es_diff_slope_L1_ME11a_even_(0),
      es_diff_slope_L1_ME11a_odd_(0),
      es_diff_slope_L1_ME11b_even_(0),
      es_diff_slope_L1_ME11b_odd_(0),
      es_diff_slope_L2_ME11a_even_(0),
      es_diff_slope_L2_ME11a_odd_(0),
      es_diff_slope_L2_ME11b_even_(0),
      es_diff_slope_L2_ME11b_odd_(0) {}

// GEM-CSC trigger: coordinate conversion
void CSCL1TPLookupTableME11ILT::set_GEM_pad_CSC_es_ME11b_even(t_lut lut) {
  GEM_pad_CSC_es_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_GEM_pad_CSC_es_ME11a_even(t_lut lut) {
  GEM_pad_CSC_es_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_GEM_pad_CSC_es_ME11b_odd(t_lut lut) { GEM_pad_CSC_es_ME11b_odd_ = std::move(lut); }

void CSCL1TPLookupTableME11ILT::set_GEM_pad_CSC_es_ME11a_odd(t_lut lut) { GEM_pad_CSC_es_ME11a_odd_ = std::move(lut); }

void CSCL1TPLookupTableME11ILT::set_GEM_roll_CSC_min_wg_ME11_even(t_lut lut) {
  GEM_roll_CSC_min_wg_ME11_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_GEM_roll_CSC_max_wg_ME11_even(t_lut lut) {
  GEM_roll_CSC_max_wg_ME11_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_GEM_roll_CSC_min_wg_ME11_odd(t_lut lut) {
  GEM_roll_CSC_min_wg_ME11_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_GEM_roll_CSC_max_wg_ME11_odd(t_lut lut) {
  GEM_roll_CSC_max_wg_ME11_odd_ = std::move(lut);
}

// GEM-CSC trigger: slope correction
void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_2to1_L1_ME11a_even(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_2to1_L1_ME11b_even(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_2to1_L1_ME11a_odd(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_2to1_L1_ME11b_odd(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_3to1_L1_ME11a_even(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_3to1_L1_ME11b_even(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_3to1_L1_ME11a_odd(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_3to1_L1_ME11b_odd(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_corr_L1_ME11a_even(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_corr_L1_ME11b_even(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_corr_L1_ME11a_odd(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_cosi_corr_L1_ME11b_odd(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L1_ME11a_even(t_lut lut) {
  CSC_slope_corr_L1_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L1_ME11b_even(t_lut lut) {
  CSC_slope_corr_L1_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L1_ME11a_odd(t_lut lut) {
  CSC_slope_corr_L1_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L1_ME11b_odd(t_lut lut) {
  CSC_slope_corr_L1_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L1_ME11a_even(t_lut lut) {
  es_diff_slope_L1_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L1_ME11a_odd(t_lut lut) {
  es_diff_slope_L1_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L1_ME11b_even(t_lut lut) {
  es_diff_slope_L1_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L1_ME11b_odd(t_lut lut) {
  es_diff_slope_L1_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L2_ME11a_even(t_lut lut) {
  CSC_slope_corr_L2_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L2_ME11b_even(t_lut lut) {
  CSC_slope_corr_L2_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L2_ME11a_odd(t_lut lut) {
  CSC_slope_corr_L2_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_CSC_slope_corr_L2_ME11b_odd(t_lut lut) {
  CSC_slope_corr_L2_ME11b_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L2_ME11a_even(t_lut lut) {
  es_diff_slope_L2_ME11a_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L2_ME11a_odd(t_lut lut) {
  es_diff_slope_L2_ME11a_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L2_ME11b_even(t_lut lut) {
  es_diff_slope_L2_ME11b_even_ = std::move(lut);
}

void CSCL1TPLookupTableME11ILT::set_es_diff_slope_L2_ME11b_odd(t_lut lut) {
  es_diff_slope_L2_ME11b_odd_ = std::move(lut);
}

// GEM-CSC trigger: coordinate conversion
unsigned CSCL1TPLookupTableME11ILT::GEM_pad_CSC_es_ME11b_even(unsigned pad) const {
  return GEM_pad_CSC_es_ME11b_even_.at(pad);
}

unsigned CSCL1TPLookupTableME11ILT::GEM_pad_CSC_es_ME11a_even(unsigned pad) const {
  return GEM_pad_CSC_es_ME11a_even_.at(pad);
}

unsigned CSCL1TPLookupTableME11ILT::GEM_pad_CSC_es_ME11b_odd(unsigned pad) const {
  return GEM_pad_CSC_es_ME11b_odd_.at(pad);
}

unsigned CSCL1TPLookupTableME11ILT::GEM_pad_CSC_es_ME11a_odd(unsigned pad) const {
  return GEM_pad_CSC_es_ME11a_odd_.at(pad);
}

unsigned CSCL1TPLookupTableME11ILT::GEM_roll_CSC_min_wg_ME11_even(unsigned roll) const {
  return GEM_roll_CSC_min_wg_ME11_even_[roll];
}

unsigned CSCL1TPLookupTableME11ILT::GEM_roll_CSC_max_wg_ME11_even(unsigned roll) const {
  return GEM_roll_CSC_max_wg_ME11_even_[roll];
}

unsigned CSCL1TPLookupTableME11ILT::GEM_roll_CSC_min_wg_ME11_odd(unsigned roll) const {
  return GEM_roll_CSC_min_wg_ME11_odd_[roll];
}

unsigned CSCL1TPLookupTableME11ILT::GEM_roll_CSC_max_wg_ME11_odd(unsigned roll) const {
  return GEM_roll_CSC_max_wg_ME11_odd_[roll];
}

// GEM-CSC trigger: slope correction
unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_2to1_L1_ME11a_even(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME11a_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_2to1_L1_ME11b_even(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME11b_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_2to1_L1_ME11a_odd(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME11a_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_2to1_L1_ME11b_odd(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME11b_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_3to1_L1_ME11a_even(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME11a_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_3to1_L1_ME11b_even(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME11b_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_3to1_L1_ME11a_odd(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME11a_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_3to1_L1_ME11b_odd(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME11b_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_corr_L1_ME11a_even(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME11a_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_corr_L1_ME11b_even(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME11b_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_corr_L1_ME11a_odd(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME11a_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_cosi_corr_L1_ME11b_odd(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME11b_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L1_ME11a_even(unsigned slope) const {
  return CSC_slope_corr_L1_ME11a_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L1_ME11b_even(unsigned slope) const {
  return CSC_slope_corr_L1_ME11b_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L1_ME11a_odd(unsigned slope) const {
  return CSC_slope_corr_L1_ME11a_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L1_ME11b_odd(unsigned slope) const {
  return CSC_slope_corr_L1_ME11b_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L1_ME11a_even(unsigned es_diff) const {
  return es_diff_slope_L1_ME11a_even_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L1_ME11a_odd(unsigned es_diff) const {
  return es_diff_slope_L1_ME11a_odd_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L1_ME11b_even(unsigned es_diff) const {
  return es_diff_slope_L1_ME11b_even_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L1_ME11b_odd(unsigned es_diff) const {
  return es_diff_slope_L1_ME11b_odd_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L2_ME11a_even(unsigned slope) const {
  return CSC_slope_corr_L2_ME11a_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L2_ME11b_even(unsigned slope) const {
  return CSC_slope_corr_L2_ME11b_even_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L2_ME11a_odd(unsigned slope) const {
  return CSC_slope_corr_L2_ME11a_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::CSC_slope_corr_L2_ME11b_odd(unsigned slope) const {
  return CSC_slope_corr_L2_ME11b_odd_[slope];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L2_ME11a_even(unsigned es_diff) const {
  return es_diff_slope_L2_ME11a_even_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L2_ME11a_odd(unsigned es_diff) const {
  return es_diff_slope_L2_ME11a_odd_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L2_ME11b_even(unsigned es_diff) const {
  return es_diff_slope_L2_ME11b_even_[es_diff];
}

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_L2_ME11b_odd(unsigned es_diff) const {
  return es_diff_slope_L2_ME11b_odd_[es_diff];
}
