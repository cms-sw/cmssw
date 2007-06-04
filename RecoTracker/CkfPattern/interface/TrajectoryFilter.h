#ifndef TrajectoryFilter_H
#define TrajectoryFilter_H

#include <string>

class Trajectory;
class TempTrajectory;

/** An abstract base class for Filter<TempTrajectory>.
 *  Adds a name() method.
 *  This class is useful because the CkfTrajectoryBuilder
 *  uses TrajectoryFilters as stopping conditions.
 */

class TrajectoryFilter {
 public:
  virtual ~TrajectoryFilter() {}
  virtual bool operator()( const TempTrajectory&) const = 0;
  virtual bool operator()( const Trajectory&) const = 0;
  virtual std::string name() const = 0;
};


#endif
