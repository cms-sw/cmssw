#ifndef TrackerGeomBuilder_TrackerHierarchy
#define TrackerGeomBuilder_TrackerHierarchy

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include<string>


// return a string describing trakcer geometry hierarchy
std::string trackerHierarchy(const TrackerTopology *tTopo, unsigned int id);

#endif // TrackerGeomBuilder_TrackerHierarchy
