#ifndef _TRACKER_MU_HIC_PROPAGATOR_H_
#define _TRACKER_MU_HIC_PROPAGATOR_H_
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "TrackingTools/MaterialEffects/interface/MaterialEffectsUpdator.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <string>
#include <iostream>
namespace cms {
class HICMuonPropagator:public Propagator{
public:
  HICMuonPropagator(const MagneticField * mf){field = mf;}
  virtual  ~HICMuonPropagator(){}
  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts,
                                     const Cylinder& cylin) const;

  TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts,
                                     const Plane& plane) const;

  void setHICConst(HICConst* hh) {theHICConst = hh;}
 
  virtual HICMuonPropagator * clone() const 
  {
    return new HICMuonPropagator(field);
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
  HICConst*             theHICConst;  
  const MagneticField * field;
};
}
#endif



