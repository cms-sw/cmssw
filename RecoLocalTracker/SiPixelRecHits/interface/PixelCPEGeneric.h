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

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"

// The template header files
//#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"

#include <utility>
#include <vector>

#if 0
/** \class PixelCPEGeneric
 * Perform the position and error evaluation of pixel hits using
 * the Det angle to estimate the track impact angle
 */
#endif

class MagneticField;
class PixelCPEGeneric final : public PixelCPEGenericBase {
public:
  PixelCPEGeneric(edm::ParameterSet const &conf,
                  const MagneticField *,
                  const TrackerGeometry &,
                  const TrackerTopology &,
                  const SiPixelLorentzAngle *,
                  const SiPixelGenErrorDBObject *,
                  const SiPixelLorentzAngle *);

  ~PixelCPEGeneric() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

private:
  LocalPoint localPosition(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;
  LocalError localError(DetParam const &theDetParam, ClusterParam &theClusterParam) const override;

  //--------------------------------------------------------------------
  //  Methods.
  //------------------------------------------------------------------

  //--- Errors squared in x and y.  &&& Need to be revisited.
  float err2X(bool &, int &) const;
  float err2Y(bool &, int &) const;

  //--- Cuts made externally settable
  float the_eff_charge_cut_lowX;
  float the_eff_charge_cut_lowY;
  float the_eff_charge_cut_highX;
  float the_eff_charge_cut_highY;
  float the_size_cutX;
  float the_size_cutY;

  bool inflate_errors;
  bool inflate_all_errors_no_trk_angle;

  bool DoCosmics_;
  bool IrradiationBiasCorrection_;
  bool isUpgrade_;
  bool NoTemplateErrorsWhenNoTrkAngles_;

  //--- DB Error Parametrization object, new light templates
  std::vector<SiPixelGenErrorStore> thePixelGenError_;
  //SiPixelCPEGenericDBErrorParametrization * genErrorsFromDB_;
};

#endif
