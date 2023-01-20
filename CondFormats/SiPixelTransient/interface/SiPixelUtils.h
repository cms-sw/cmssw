#ifndef CondFormats_SiPixelTransient_SiPixelUtils_h
#define CondFormats_SiPixelTransient_SiPixelUtils_h

namespace siPixelUtils {
  float generic_position_formula(int size,                    //!< Size of this projection.
                                 int q_f,                     //!< Charge in the first pixel.
                                 int q_l,                     //!< Charge in the last pixel.
                                 float upper_edge_first_pix,  //!< As the name says.
                                 float lower_edge_last_pix,   //!< As the name says.
                                 float lorentz_shift,         //!< L-width
                                 float theThickness,          //detector thickness
                                 float cot_angle,             //!< cot of alpha_ or beta_
                                 float pitch,                 //!< thePitchX or thePitchY
                                 bool first_is_big,           //!< true if the first is big
                                 bool last_is_big,            //!< true if the last is big
                                 float eff_charge_cut_low,    //!< Use edge if > W_eff (in pix) &&&
                                 float eff_charge_cut_high,   //!< Use edge if < W_eff (in pix) &&&
                                 float size_cut               //!< Use edge when size == cuts
  );
}  // namespace siPixelUtils

#endif
