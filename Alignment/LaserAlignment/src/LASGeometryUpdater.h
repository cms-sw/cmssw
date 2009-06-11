
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
#include "Alignment/LaserAlignment/src/LASConstants.h"


class LASGeometryUpdater {
  
 public:
  LASGeometryUpdater( LASGlobalData<LASCoordinateSet>&, LASConstants& );
  void ApplyBeamKinkCorrections( LASGlobalData<LASCoordinateSet>& ) const;
  void EndcapUpdate( LASEndcapAlignmentParameterSet&, LASGlobalData<LASCoordinateSet>& );
  void TrackerUpdate( LASEndcapAlignmentParameterSet&, LASBarrelAlignmentParameterSet&, AlignableTracker& );
  void SetReverseDirection( bool );
  void SetMisalignmentFromRefGeometry( bool );

 private:
  LASGlobalData<LASCoordinateSet> nominalCoordinates;
  LASConstants lasConstants;
  bool isReverseDirection;
  bool isMisalignmentFromRefGeometry;

};

#endif
