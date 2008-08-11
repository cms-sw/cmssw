#ifndef _TRACKER_FASTMUPROPAGATOR_H_
#define _TRACKER_FASTMUPROPAGATOR_H_
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoHIMuon/HiMuPropagator/interface/FmpConst.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <string>
#include <iostream>
#include <map>
#include <vector>

/** A very fast propagator that can go only from the
 *  trigger layers of the muon system to the tracker bounds.
 *  Uses precomputed parametrizations.
 */

class FastMuPropagator:public Propagator{
public:
  FastMuPropagator(const MagneticField * mf, PropagationDirection dir = alongMomentum)
                                    {theFmpConst=new FmpConst(); field = mf;} 
                        

  virtual  ~FastMuPropagator() {delete theFmpConst;}

  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts,
				     const Cylinder& bound) const;
				      
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                     const Plane&) const;

  virtual FastMuPropagator * clone() const 
  {
  PropagationDirection dir = alongMomentum;  
  return new FastMuPropagator(field,dir);
  } 
   
  TrajectoryStateOnSurface propagate (const FreeTrajectoryState& fts,
                                      const Surface& surface) const{
    return Propagator::propagate( fts, surface);
  }
  
  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState& state, const Plane& bc) const {
  std::pair<TrajectoryStateOnSurface,double> tp;
    return tp;
  }
  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState& state, const Cylinder& bc) const{
  std::pair<TrajectoryStateOnSurface,double> tp;
    return tp;
  }
  
  virtual const MagneticField* magneticField() const {return field;}
    
private:
  bool checkfts(const FreeTrajectoryState& fts) const;
  FmpConst* theFmpConst;
  const MagneticField * field;	      					      
};

#endif



