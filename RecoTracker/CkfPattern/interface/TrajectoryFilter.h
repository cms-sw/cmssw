#ifndef TrajectoryFilter_H
#define TrajectoryFilter_H

#include <string>

class Trajectory;

/** An abstract base class for Filter<Trajectory>.
 *  Adds a name() method.
 *  This class is useful because the CkfTrajectoryBuilder
 *  uses TrajectoryFilters as stopping conditions.
 */

class TrajectoryFilter{
 public:
  virtual ~TrajectoryFilter() {}
  virtual bool operator()( const Trajectory&) const = 0;
  virtual std::string name() const = 0;
};

#endif
