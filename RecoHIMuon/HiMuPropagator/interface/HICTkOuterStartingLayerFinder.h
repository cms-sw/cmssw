#ifndef _HICTKOUTERSTARTINGLAYERFINDER_H_ 
#define _HICTKOUTERSTARTINGLAYERFINDER_H_

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"


#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include <vector>


/** Finds the layers that a FreeTrajectoryState which is outside
 *  of the tracker volume crosses first.
 *  In the forward and barrel-forward regions there may be
 *  several such layers.
 */

namespace cms{
class HICTkOuterStartingLayerFinder {
public:
  typedef std::vector<DetLayer*>                              LayerContainer;
  
  HICTkOuterStartingLayerFinder(int&, const MagneticField * mf, const GeometricSearchTracker* th, const HICConst* );

  ~HICTkOuterStartingLayerFinder(){};
    
  LayerContainer startingLayers(FreeTrajectoryState& fts);
 
	     
private:

  bool findForwardLayers(const FreeTrajectoryState& fts, 
          std::vector<ForwardDetLayer*>& fls, LayerContainer& lc); 
  LayerContainer findBarrelLayers(const FreeTrajectoryState& fts, 
          std::vector<ForwardDetLayer*>& fls, LayerContainer& lc);      
  
  std::vector<BarrelDetLayer*>               theBarrelLayers;
  std::vector<ForwardDetLayer*>              forwardPosLayers;
  std::vector<ForwardDetLayer*>              forwardNegLayers;
  std::vector<DetLayer*>                     theDetLayer;
  const MagneticField*                       magfield;
  const GeometricSearchTracker*              theTracker;
  int                                        NumberOfSigm;
  const HICConst*                            theHICConst;
  double                                     length; 
};
}
#endif





