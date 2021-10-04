#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelTransient/interface/SiPixelUtils.h"
#else
#include "SiPixelUtils.h"
#endif

#include <cmath>

namespace SiPixelUtils {

  //-----------------------------------------------------------------------------
  //!  A generic version of the position formula.  Since it works for both
  //!  X and Y, in the interest of the simplicity of the code, all parameters
  //!  are passed by the caller.
  //-----------------------------------------------------------------------------
  float generic_position_formula(int size,                    //!< Size of this projection.
                                 int q_f,                     //!< Charge in the first pixel.
                                 int q_l,                     //!< Charge in the last pixel.
                                 float upper_edge_first_pix,  //!< As the name says.
                                 float lower_edge_last_pix,   //!< As the name says.
                                 float lorentz_shift,         //!< L-shift at half thickness
                                 float theThickness,          //detector thickness
                                 float cot_angle,             //!< cot of alpha_ or beta_
                                 float pitch,                 //!< thePitchX or thePitchY
                                 bool first_is_big,           //!< true if the first is big
                                 bool last_is_big,            //!< true if the last is big
                                 float eff_charge_cut_low,    //!< Use edge if > w_eff  &&&
                                 float eff_charge_cut_high,   //!< Use edge if < w_eff  &&&
                                 float size_cut               //!< Use edge when size == cuts
  ) {
    float geom_center = 0.5f * (upper_edge_first_pix + lower_edge_last_pix);

    //--- The case of only one pixel in this projection is separate.  Note that
    //--- here first_pix == last_pix, so the average of the two is still the
    //--- center of the pixel.
    if (size == 1) {
      return geom_center;
    }

    //--- Width of the clusters minus the edge (first and last) pixels.
    //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
    float w_inner = lower_edge_last_pix - upper_edge_first_pix;  // in cm

    //--- Predicted charge width from geometry
    float w_pred = theThickness * cot_angle  // geometric correction (in cm)
                   - lorentz_shift;          // (in cm) &&& check fpix!

    //--- Total length of the two edge pixels (first+last)
    float sum_of_edge = 2.0f;
    if (first_is_big)
      sum_of_edge += 1.0f;
    if (last_is_big)
      sum_of_edge += 1.0f;

    //--- The `effective' charge width -- particle's path in first and last pixels only
    float w_eff = std::abs(w_pred) - w_inner;

    //--- If the observed charge width is inconsistent with the expectations
    //--- based on the track, do *not* use w_pred-w_innner.  Instead, replace
    //--- it with an *average* effective charge width, which is the average
    //--- length of the edge pixels.
    //
    //  bool usedEdgeAlgo = false;
    if ((size >= size_cut) || ((w_eff / pitch < eff_charge_cut_low) | (w_eff / pitch > eff_charge_cut_high))) {
      w_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
                                           //  usedEdgeAlgo = true;
    }

    //--- Finally, compute the position in this projection
    float q_diff = q_l - q_f;
    float q_sum = q_l + q_f;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (q_sum == 0)
      q_sum = 1.0f;

    float hit_pos = geom_center + 0.5f * (q_diff / q_sum) * w_eff;

    return hit_pos;
  }

  float generic_position_formula_y_bricked(
      int size,                    //!< Size of this projection.
      int q_f,                     //!< Charge in the first pixel.
      int q_l,                     //!< Charge in the last pixel.
      int q_f_b,                   //!< Charge in pixels that are "dented" compared to the lowest pixel of the cluster.
      int q_l_b,                   //!< Charge in pixels that are "dented" compared to the highest pixel of the cluster.
      float upper_edge_first_pix,  //!< As the name says.
      float lower_edge_last_pix,   //!< As the name says.
      float lorentz_shift,         //!< L-shift at half thickness
      float theThickness,          //detector thickness
      float cot_angle,             //!< cot of alpha_ or beta_
      float pitch,                 //!< thePitchX or thePitchY
      bool first_is_big,           //!< true if the first is big
      bool last_is_big,            //!< true if the last is big
      float eff_charge_cut_low,    //!< Use edge if > w_eff  &&&
      float eff_charge_cut_high,   //!< Use edge if < w_eff  &&&
      float size_cut               //!< Use edge when size == cuts
  ) {
    const auto geom_center = 0.5f * (upper_edge_first_pix + lower_edge_last_pix);

    //--- The case of only one pixel in this projection is separate.  Note that
    //--- here first_pix == last_pix, so the average of the two is still the
    //--- center of the pixel.

    //--- Width of the clusters minus the edge (first and last) pixels.
    //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
    const auto w_inner = lower_edge_last_pix - upper_edge_first_pix;  // in cm

    //--- Predicted charge width from geometry
    const auto w_pred = theThickness * cot_angle  // geometric correction (in cm)
                        - lorentz_shift;          // (in cm) &&& check fpix!

    //--- Total length of the two edge pixels (first+last)
    auto sum_of_edge = 2.0f;
    if (first_is_big)
      sum_of_edge += 1.0f;
    if (last_is_big)
      sum_of_edge += 1.0f;

    //--- The `effective' charge width -- particle's path in first and last pixels only
    auto w_eff = std::abs(w_pred) - std::abs(w_inner);

    //--- If the observed charge width is inconsistent with the expectations
    //--- based on the track, do *not* use w_pred-w_innner.  Instead, replace
    //--- it with an *average* effective charge width, which is the average
    //--- length of the edge pixels.
    //
    //  bool usedEdgeAlgo = false;
    //Modified cut to make use of the w_eff in the bricked geometry
    if (size >= size_cut) {
      w_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
                                           //  usedEdgeAlgo = true;
    }

    //--- Finally, compute the position in this projection
    const auto q_diff = q_l - q_f;
    auto q_sum = q_l + q_f;
    const auto q_b_corr = q_l_b + q_f_b;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (q_sum == 0)
      q_sum = 1.0f;

    float hit_pos =
        geom_center + 0.5f * (q_diff / q_sum) * w_eff + 0.5f * (q_b_corr / q_sum) * w_eff;  //bricked correction

    return hit_pos;
  }
}  // namespace SiPixelUtils
