
#ifndef __LASGEOMETRYUPDATER_H
#define __LASGEOMETRYUPDATER_H

#include <vector>
#include <cmath>

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/LaserAlignment/src/LASEndcapAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASGlobalData.cc" //template
#include "Alignment/LaserAlignment/src/LASCoordinateSet.h" 
#include "Alignment/LaserAlignment/src/LASGlobalLoop.h"


class LASGeometryUpdater {
  
 public:
  LASGeometryUpdater( LASGlobalData<LASCoordinateSet>& );
  void EndcapUpdate( LASEndcapAlignmentParameterSet&, LASGlobalData<LASCoordinateSet>& );
  void TrackerUpdate( LASEndcapAlignmentParameterSet&, LASBarrelAlignmentParameterSet&, AlignableTracker& );

 private:
  LASGlobalData<LASCoordinateSet> nominalCoordinates;


};

#endif
