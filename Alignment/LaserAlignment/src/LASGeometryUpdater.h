
#ifndef __LASGEOMETRYUPDATER_H
#define __LASGEOMETRYUPDATER_H

#include <vector>

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/LaserAlignment/src/LASEndcapAlignmentParameterSet.h"
#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"

class LASGeometryUpdater {
  
 public:
  LASGeometryUpdater();
  void Update( LASEndcapAlignmentParameterSet&, LASBarrelAlignmentParameterSet&, AlignableTracker& );

 private:
  


};

#endif
