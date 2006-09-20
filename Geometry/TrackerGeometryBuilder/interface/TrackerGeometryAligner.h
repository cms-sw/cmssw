#ifndef Geometry_TrackerGeometryBuilder_TrackerGeometryAligner_H
#define Geometry_TrackerGeometryBuilder_TrackerGeometryAligner_H

#include <vector>

#include "Geometry/CommonDetUnit/interface/DetPositioner.h"

class Alignments;


/// Class to update the tracker geometry with a set of alignments

class TrackerGeometryAligner : public DetPositioner {

public:
  void applyAlignments( TrackerGeometry* tracker, const Alignments* alignments );

};

#endif
