#ifndef TrackerGeometryBuilder_TrackerParametersFromDD_h
#define TrackerGeometryBuilder_TrackerParametersFromDD_h

#include <vector>

class DDCompactView;
class PTrackerParameters;

class TrackerParametersFromDD {
 public:
  TrackerParametersFromDD() {}
  virtual ~TrackerParametersFromDD() {}

  bool build( const DDCompactView*,
	      PTrackerParameters& );
 private:
  void putOne( int, std::vector<int> &, PTrackerParameters& );
};

#endif
