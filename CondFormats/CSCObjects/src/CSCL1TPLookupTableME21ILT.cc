#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME21ILT.h"

CSCL1TPLookupTableME21ILT::CSCL1TPLookupTableME21ILT()
    : GEM_pad_CSC_es_ME21_even_(0),
      GEM_pad_CSC_es_ME21_odd_(0),

      GEM_roll_L1_CSC_min_wg_ME21_even_(0),
      GEM_roll_L1_CSC_max_wg_ME21_even_(0),
      GEM_roll_L1_CSC_min_wg_ME21_odd_(0),
      GEM_roll_L1_CSC_max_wg_ME21_odd_(0),

      GEM_roll_L2_CSC_min_wg_ME21_even_(0),
      GEM_roll_L2_CSC_max_wg_ME21_even_(0),
      GEM_roll_L2_CSC_min_wg_ME21_odd_(0),
      GEM_roll_L2_CSC_max_wg_ME21_odd_(0),

      CSC_slope_cosi_2to1_L1_ME21_even_(0),
      CSC_slope_cosi_2to1_L1_ME21_odd_(0),
      CSC_slope_cosi_3to1_L1_ME21_even_(0),
      CSC_slope_cosi_3to1_L1_ME21_odd_(0),

      CSC_slope_cosi_corr_L1_ME21_even_(0),
      CSC_slope_cosi_corr_L1_ME21_odd_(0),

      CSC_slope_corr_L1_ME21_even_(0),
      CSC_slope_corr_L1_ME21_odd_(0),
      CSC_slope_corr_L2_ME21_even_(0),
      CSC_slope_corr_L2_ME21_odd_(0),

      es_diff_slope_L1_ME21_even_(0),
      es_diff_slope_L1_ME21_odd_(0),
      es_diff_slope_L2_ME21_even_(0),
      es_diff_slope_L2_ME21_odd_(0) {}

void CSCL1TPLookupTableME21ILT::set_GEM_pad_CSC_es_ME21_even(t_lut lut) { GEM_pad_CSC_es_ME21_even_ = std::move(lut); }

void CSCL1TPLookupTableME21ILT::set_GEM_pad_CSC_es_ME21_odd(t_lut lut) { GEM_pad_CSC_es_ME21_odd_ = std::move(lut); }

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L1_CSC_min_wg_ME21_even(t_lut lut) {
  GEM_roll_L1_CSC_min_wg_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L1_CSC_max_wg_ME21_even(t_lut lut) {
  GEM_roll_L1_CSC_max_wg_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L1_CSC_min_wg_ME21_odd(t_lut lut) {
  GEM_roll_L1_CSC_min_wg_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L1_CSC_max_wg_ME21_odd(t_lut lut) {
  GEM_roll_L1_CSC_max_wg_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L2_CSC_min_wg_ME21_even(t_lut lut) {
  GEM_roll_L2_CSC_min_wg_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L2_CSC_max_wg_ME21_even(t_lut lut) {
  GEM_roll_L2_CSC_max_wg_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L2_CSC_min_wg_ME21_odd(t_lut lut) {
  GEM_roll_L2_CSC_min_wg_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_GEM_roll_L2_CSC_max_wg_ME21_odd(t_lut lut) {
  GEM_roll_L2_CSC_max_wg_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_2to1_L1_ME21_even(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_2to1_L1_ME21_odd(t_lut lut) {
  CSC_slope_cosi_2to1_L1_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_3to1_L1_ME21_even(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_3to1_L1_ME21_odd(t_lut lut) {
  CSC_slope_cosi_3to1_L1_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_corr_L1_ME21_even(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_cosi_corr_L1_ME21_odd(t_lut lut) {
  CSC_slope_cosi_corr_L1_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_corr_L1_ME21_even(t_lut lut) {
  CSC_slope_corr_L1_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_corr_L1_ME21_odd(t_lut lut) {
  CSC_slope_corr_L1_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_es_diff_slope_L1_ME21_even(t_lut lut) {
  es_diff_slope_L1_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_es_diff_slope_L1_ME21_odd(t_lut lut) {
  es_diff_slope_L1_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_corr_L2_ME21_even(t_lut lut) {
  CSC_slope_corr_L2_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_CSC_slope_corr_L2_ME21_odd(t_lut lut) {
  CSC_slope_corr_L2_ME21_odd_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_es_diff_slope_L2_ME21_even(t_lut lut) {
  es_diff_slope_L2_ME21_even_ = std::move(lut);
}

void CSCL1TPLookupTableME21ILT::set_es_diff_slope_L2_ME21_odd(t_lut lut) {
  es_diff_slope_L2_ME21_odd_ = std::move(lut);
}

unsigned CSCL1TPLookupTableME21ILT::GEM_pad_CSC_es_ME21_even(unsigned pad) const {
  return GEM_pad_CSC_es_ME21_even_[pad];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_pad_CSC_es_ME21_odd(unsigned pad) const {
  return GEM_pad_CSC_es_ME21_odd_[pad];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L1_CSC_min_wg_ME21_even(unsigned roll) const {
  return GEM_roll_L1_CSC_min_wg_ME21_even_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L1_CSC_max_wg_ME21_even(unsigned roll) const {
  return GEM_roll_L1_CSC_max_wg_ME21_even_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L1_CSC_min_wg_ME21_odd(unsigned roll) const {
  return GEM_roll_L1_CSC_min_wg_ME21_odd_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L1_CSC_max_wg_ME21_odd(unsigned roll) const {
  return GEM_roll_L1_CSC_max_wg_ME21_odd_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L2_CSC_min_wg_ME21_even(unsigned roll) const {
  return GEM_roll_L2_CSC_min_wg_ME21_even_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L2_CSC_max_wg_ME21_even(unsigned roll) const {
  return GEM_roll_L2_CSC_max_wg_ME21_even_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L2_CSC_min_wg_ME21_odd(unsigned roll) const {
  return GEM_roll_L2_CSC_min_wg_ME21_odd_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::GEM_roll_L2_CSC_max_wg_ME21_odd(unsigned roll) const {
  return GEM_roll_L2_CSC_max_wg_ME21_odd_[roll];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_2to1_L1_ME21_even(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME21_even_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_2to1_L1_ME21_odd(unsigned slope) const {
  return CSC_slope_cosi_2to1_L1_ME21_odd_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_3to1_L1_ME21_even(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME21_even_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_3to1_L1_ME21_odd(unsigned slope) const {
  return CSC_slope_cosi_3to1_L1_ME21_odd_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_corr_L1_ME21_even(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME21_even_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_cosi_corr_L1_ME21_odd(unsigned slope) const {
  return CSC_slope_cosi_corr_L1_ME21_odd_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_corr_L1_ME21_even(unsigned slope) const {
  return CSC_slope_corr_L1_ME21_even_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_corr_L1_ME21_odd(unsigned slope) const {
  return CSC_slope_corr_L1_ME21_odd_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::es_diff_slope_L1_ME21_even(unsigned es_diff) const {
  return es_diff_slope_L1_ME21_even_[es_diff];
}

unsigned CSCL1TPLookupTableME21ILT::es_diff_slope_L1_ME21_odd(unsigned es_diff) const {
  return es_diff_slope_L1_ME21_odd_[es_diff];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_corr_L2_ME21_even(unsigned slope) const {
  return CSC_slope_corr_L2_ME21_even_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::CSC_slope_corr_L2_ME21_odd(unsigned slope) const {
  return CSC_slope_corr_L2_ME21_odd_[slope];
}

unsigned CSCL1TPLookupTableME21ILT::es_diff_slope_L2_ME21_even(unsigned es_diff) const {
  return es_diff_slope_L2_ME21_even_[es_diff];
}

unsigned CSCL1TPLookupTableME21ILT::es_diff_slope_L2_ME21_odd(unsigned es_diff) const {
  return es_diff_slope_L2_ME21_odd_[es_diff];
}
