#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEGenericForBricked_H
#define RecoLocalTracker_SiPixelRecHits_PixelCPEGenericForBricked_H

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

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"

// The template header files
//#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"

#include <utility>
#include <vector>

#if 0
/** \class PixelCPEGenericForBricked
 * Perform the position and error evaluation of pixel hits using
 * the Det angle to estimate the track impact angle
 */
#endif

class MagneticField;
class PixelCPEGenericForBricked final : public PixelCPEGeneric {
public:
  PixelCPEGenericForBricked(edm::ParameterSet const& conf,
                            const MagneticField*,
                            const TrackerGeometry&,
                            const TrackerTopology&,
                            const SiPixelLorentzAngle*,
                            const SiPixelGenErrorDBObject*,
                            const SiPixelLorentzAngle*);

  ~PixelCPEGenericForBricked() override{};

private:
  LocalPoint localPosition(DetParam const& theDetParam, ClusterParam& theClusterParam) const override;
  static void collect_edge_charges_bricked(ClusterParam& theClusterParam,  //!< input, the cluster
                                           int& q_f_X,                     //!< output, Q first  in X
                                           int& q_l_X,                     //!< output, Q last   in X
                                           int& q_f_Y,                     //!< output, Q first  in Y
                                           int& q_l_Y,                     //!< output, Q last   in Y
                                           int& Q_f_b,
                                           int& Q_l_b,               //Bricked correction
                                           int& lowest_is_bricked,   //Bricked correction
                                           int& highest_is_bricked,  //Bricked correction
                                           bool truncate);
};

#endif
