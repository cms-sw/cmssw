#ifndef Alignment_LaserAlignment_TsosVectorCollection_H
#define Alignment_LaserAlignment_TsosVectorCollection_H

/**
   \author Gero Flucke 
   \date May 2009
   last update on $Date: 2009/10/14 07:32:07 $ by $Author: flucke $
   
   A typedef for a collection of std::vectors of TrajectoryStateOnSurface .
   
*/


#include <vector>
// class TrajectoryStateOnSurface; // sufficient, but also confusing, so include:
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
typedef std::vector<std::vector<TrajectoryStateOnSurface> > TsosVectorCollection;

#endif
