
#ifndef __LASGEOMETRYUPDATER_H
#define __LASGEOMETRYUPDATER_H

#include <vector>

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/LaserAlignment/src/LASEndcapAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASGlobalData.cc" //template
#include "Alignment/LaserAlignment/src/LASCoordinateSet.h" 


class LASGeometryUpdater {
  
 public:
  LASGeometryUpdater();
  void EndcapUpdate( LASEndcapAlignmentParameterSet&, LASGlobalData<LASCoordinateSet>&, AlignableTracker& );
  void TrackerUpdate( LASEndcapAlignmentParameterSet&, LASBarrelAlignmentParameterSet&, AlignableTracker& );

 private:
  


};

#endif
