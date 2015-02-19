#ifndef TrackerGeometryBuilder_TrackerParametersFromDD_h
#define TrackerGeometryBuilder_TrackerParametersFromDD_h

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class DDCompactView;
class PTrackerParameters;

class TrackerParametersFromDD {
 public:
  TrackerParametersFromDD() {}
  virtual ~TrackerParametersFromDD() {}

  bool build( const DDCompactView*,
	      PTrackerParameters& );
 private:
  void putOne( GeometricDet::GeometricEnumType, std::vector<int> &, PTrackerParameters& );
};

#endif
