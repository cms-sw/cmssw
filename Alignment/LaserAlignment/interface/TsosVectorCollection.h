/**
   \author Gero Flucke 
   \date May 2009
   last update on $Date$ by $Author$
   
   A typedef for a collection of std::vectors of TrajectoryStateOnSurface .
   
*/


#ifndef Alignment_LaserAlignment_TsosVectorCollection_H
#define Alignment_LaserAlignment_TsosVectorCollection_H

#include <vector>
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

typedef std::vector<std::vector<TrajectoryStateOnSurface> > TsosVectorCollection;

#endif
