#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEGeneric_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEGeneric_H

// \class PixelCPEGeneric  -- a generalized CPE reco for the idealized detector
//
// The basic idea of this class is to use generic formulae in order
// to achieve clean and minimal code.  It should work for
// - both normal and big pixels
// - both barrel and forward
// - both "FromDetPosition" and "FromTrackAngles" (i.e. by the track fit)
//
// This is possible since, in its nature, the original "ORCA" algorithm by 
// Danek and Susana is the same in both X and Y directions, provided that
// one correctly computes angles alpha_ and beta_ up front.  Thus, all
// geometrical and special corrections are dropped, since the presumption
// is that alpha_ and beta_ are determined as best as possible.  That means
// that they either come from the track, or, if they come from the 
// position of the DetUnit, they include all geometrical information 
// possible for this DetUnit:
// - for both the barrel and the forward, we use the cluster position 
//   instead of the center of the module/plaquette
// - for the forward, the tilt of the blades is included too
//
// In addtion, anything which is special for the computation of the lorentz
// angle is done in setTheDet() method.  So the algorithm per se does not
// need to worry about it.  This includes extra E*B term (a.k.a. "alpha2Order")
// and extra tilt in the forward.
//
// Thus, the formula for the computation of the hit position is very
// simple, and is described in Morris's note (IN ???) on the generalizaton
// of the pixel algorithm.

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"


// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"


#include <utility>
#include <vector>


#if 0
/** \class PixelCPEGeneric
 * Perform the position and error evaluation of pixel hits using 
 * the Det angle to estimate the track impact angle 
*/
#endif

class MagneticField;
class PixelCPEGeneric : public PixelCPEBase
{
 public:
  // PixelCPEGeneric( const DetUnit& det );
  PixelCPEGeneric(edm::ParameterSet const& conf, const MagneticField *, const SiPixelLorentzAngle *, const SiPixelCPEGenericErrorParm *, const SiPixelTemplateDBObject *);
  ~PixelCPEGeneric() {;}

  LocalPoint localPosition (const SiPixelCluster& cluster, const GeomDetUnit & det) const; 
  
  // However, we do need to implement localError().
  LocalError localError   (const SiPixelCluster& cl, const GeomDetUnit & det) const;
 
 private:
  //--------------------------------------------------------------------
  //  Methods.
  //------------------------------------------------------------------
  double
    generic_position_formula( int size,                //!< Size of this projection.
			      double Q_f,              //!< Charge in the first pixel.
			      double Q_l,              //!< Charge in the last pixel.
			      double upper_edge_first_pix, //!< As the name says.
			      double lower_edge_last_pix,  //!< As the name says.
			      double half_lorentz_shift,   //!< L-shift at half thickness
			      double cot_angle,            //!< cot of alpha_ or beta_
			      double pitch,            //!< thePitchX or thePitchY
			      bool first_is_big,       //!< true if the first is big
			      bool last_is_big,        //!< true if the last is big
			      double eff_charge_cut_low, //!< Use edge if > W_eff (in pix) &&&
			      double eff_charge_cut_high,//!< Use edge if < W_eff (in pix) &&&
			      double size_cut           //!< Use edge when size == cuts
			      ) const;

  void
    collect_edge_charges(const SiPixelCluster& cluster,  //!< input, the cluster
			 float & Q_f_X,              //!< output, Q first  in X 
			 float & Q_l_X,              //!< output, Q last   in X
			 float & Q_m_X,              //!< output, Q middle in X
			 float & Q_f_Y,              //!< output, Q first  in Y 
			 float & Q_l_Y,              //!< output, Q last   in Y
			 float & Q_m_Y               //!< output, Q middle in Y
			 ) const;
  
  
  //--- Errors squared in x and y.  &&& Need to be revisited.
  float err2X(bool&, int&) const;
  float err2Y(bool&, int&) const;

  //--- Cuts made externally settable
  double the_eff_charge_cut_lowX;
  double the_eff_charge_cut_lowY;
  double the_eff_charge_cut_highX;
  double the_eff_charge_cut_highY;
  double the_size_cutX;
  double the_size_cutY;

  bool inflate_errors;
  bool inflate_all_errors_no_trk_angle;

  bool UseErrorsFromTemplates_;
  bool DoCosmics_;
  bool LoadTemplatesFromDB_;
  bool TruncatePixelCharge_;
  bool IrradiationBiasCorrection_;
  bool isUpgrade_;

  double EdgeClusterErrorX_;
  double EdgeClusterErrorY_;

  std::vector<float> xerr_barrel_l1_,yerr_barrel_l1_,xerr_barrel_ln_;
  std::vector<float> yerr_barrel_ln_,xerr_endcap_,yerr_endcap_;
  float xerr_barrel_l1_def_, yerr_barrel_l1_def_,xerr_barrel_ln_def_;
  float yerr_barrel_ln_def_, xerr_endcap_def_, yerr_endcap_def_;


  //--- DB Error Parametrization object
  SiPixelCPEGenericDBErrorParametrization * genErrorsFromDB_;

  mutable SiPixelTemplate templ_;
  mutable int templID_; 

	// The truncation value pix_maximum is an angle-dependent cutoff on the
	// individual pixel signals. It should be applied to all pixels in the
	// cluster [signal_i = fminf(signal_i, pixmax)] before the column and row
	// sums are made. Morris
	mutable float pixmx;

	// These are errors predicted by PIXELAV
	mutable float sigmay; // CPE Generic y-error for multi-pixel cluster
	mutable float sigmax; // CPE Generic x-error for multi-pixel cluster
	mutable float sy1   ; // CPE Generic y-error for single single-pixel
	mutable float sy2   ; // CPE Generic y-error for single double-pixel cluster
	mutable float sx1   ; // CPE Generic x-error for single single-pixel cluster
	mutable float sx2   ; // CPE Generic x-error for single double-pixel cluster

	// These are irradiation bias corrections
	mutable float deltay; // CPE Generic y-bias for multi-pixel cluster
	mutable float deltax; // CPE Generic x-bias for multi-pixel cluster
	mutable float dy1   ; // CPE Generic y-bias for single single-pixel cluster
	mutable float dy2   ; // CPE Generic y-bias for single double-pixel cluster
	mutable float dx1   ; // CPE Generic x-bias for single single-pixel cluster
	mutable float dx2   ; // CPE Generic x-bias for single double-pixel cluster

	 
 protected:
  //--- These functions are no longer needed, yet they are declared 
  //--- pure virtual in the base class.
  float xpos( const SiPixelCluster& ) const { return -999000.0; }  // &&& should abort
  float ypos( const SiPixelCluster& ) const { return -999000.0; }  // &&& should abort

};

#endif




