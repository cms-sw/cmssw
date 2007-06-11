#ifndef DeDxTools_H
#define DeDxTools_H
#include <vector>
#include "DataFormats/TrackReco/interface/DeDxHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class reco::Track;
namespace DeDxTools  {
 
  struct RawHits {
    double charge;
    double angleCosine;
    DetId detId;
  };
  
  std::vector<RawHits> trajectoryRawHits(const Trajectory & trajectory);
 
///////// some helper function maybe they are useless 
 
   float massFromBeta(float beta, const reco::Track &);
   
   double genericAverage(const reco::DeDxHitCollection &, float expo = 1.);

 /** 
 * Default implementation simply returns the beta obtained inverting bethe bloch 
 * for the energy obtained with dedx() 
 *  
 *  The error is computed from bethe-bloch first derivative and ...?
 *  (Landau width)/sqrt(n) ??????
 *
 */
   Measurement1D beta(const reco::TrackDeDxHits & trackWithHits);
  
  /*
   * Invert bethe bloch formula
   */
   float betaFromInvertedBetheBloch(float dedx);
 
 }

#endif
