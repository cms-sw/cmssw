#ifndef _TRACKER_HICTRAJCORRECTOR_H_
#define _TRACKER_HICTRAJCORRECTOR_H_
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

 
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Vector/ThreeVector.h"
#include <CLHEP/Vector/LorentzVector.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>

class HICTrajectoryCorrector{
public:
  HICTrajectoryCorrector(const MagneticField * mf,const HICConst* hh){field = mf;theHICConst = hh;}
  virtual  ~HICTrajectoryCorrector(){}
  TrajectoryStateOnSurface correct(FreeTrajectoryState& rh,
                                   FreeTrajectoryState& ftsnew,
                                   const GeomDet* plane) const;

  double findPhiInVertex( const FreeTrajectoryState& fts, 
                          const double& rc, 
                          const GeomDet* det ) const;  
private:
  const MagneticField * field; 
  const HICConst *      theHICConst;
};

#endif



