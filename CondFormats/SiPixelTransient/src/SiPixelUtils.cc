#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelTransient/interface/SiPixelUtils.h"
#else
#include "SiPixelUtils.h"
#endif

#include <cmath>

namespace siPixelUtils {

  //-----------------------------------------------------------------------------
  //!  A generic version of the position formula.  Since it works for both
  //!  X and Y, in the interest of the simplicity of the code, all parameters
  //!  are passed by the caller.
  //-----------------------------------------------------------------------------
  float generic_position_formula(
      int size,                    //!< Size of this projection.
      int q_f,                     //!< Charge in the first pixel.
      int q_l,                     //!< Charge in the last pixel.
      float upper_edge_first_pix,  //!< As the name says.
      float lower_edge_last_pix,   //!< As the name says.
      float lorentz_shift,         //!< L-shift at half thickness
      float theThickness,          //detector thickness
      float cot_angle,             //!< cot of alpha_ or beta_
      float pitch,                 //!< thePitchX or thePitchY
      float pitchfraction_first,
      float pitchfraction_last,
      float eff_charge_cut_low,   //!< Use edge if > w_eff  &&&
      float eff_charge_cut_high,  //!< Use edge if < w_eff  &&&
      float size_cut,             //!< Use edge when size == cuts
      float delta_length_cut,     //!< if charge len - cls size > this (in pix), use one-sided reco
      bool goodEdgeAlgo) {
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
    float sum_of_edge = pitchfraction_first + pitchfraction_last;

    //--- The `effective' charge width -- particle's path in first and last pixels only
    float w_eff = std::abs(w_pred) - w_inner;
    float delta = w_eff - 0.5 * sum_of_edge * pitch;

    //--- If the observed charge width is inconsistent with the expectations
    //--- based on the track, do *not* use w_pred-w_innner.  Instead, replace
    //--- it with an *average* effective charge width, which is the average
    //--- length of the edge pixels.
    //
    if ((size >= size_cut) || ((w_eff / pitch < eff_charge_cut_low) | (w_eff / pitch > eff_charge_cut_high))) {
      w_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)

      if (goodEdgeAlgo) {
        if (delta / pitch > delta_length_cut) {
          //  observed cluster is much shorter than expected, use one-sided reco
          if (w_pred > 0.f) {
            // x1,x2 are centers of the first last last pixel coordinates
            float x1 = upper_edge_first_pix - 0.5 * pitchfraction_first * pitch;
            float hit_pos = x1 + 0.5 * w_pred;
            return hit_pos;
          } else {
            float x2 = lower_edge_last_pix + 0.5 * pitchfraction_last * pitch;
            float hit_pos = x2 + 0.5 * w_pred;
            return hit_pos;
          }
        }  //if (delta / pitch > delta_length_cut)
      }  //if(goodEdgeAlgo)
    }  //if(size >= size_cut) ||...

    //--- Finally, compute the position in this projection
    float q_diff = q_l - q_f;
    float q_sum = q_l + q_f;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (q_sum == 0)
      q_sum = 1.0f;

    float hit_pos = geom_center + 0.5f * (q_diff / q_sum) * w_eff;

    return hit_pos;
  }
}  // namespace siPixelUtils
