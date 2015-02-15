#ifndef TrackerGeometryBuilder_TrackerParametersFromDD_h
#define TrackerGeometryBuilder_TrackerParametersFromDD_h

class DDCompactView;
class PTrackerParameters;

class TrackerParametersFromDD {
 public:
  TrackerParametersFromDD() {}
  virtual ~TrackerParametersFromDD() {}

  bool build( const DDCompactView*,
	      PTrackerParameters& );
};

#endif
