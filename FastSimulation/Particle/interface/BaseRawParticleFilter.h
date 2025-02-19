#ifndef BASERAWPARTTICLEFILTER_H
#define BASERAWPARTTICLEFILTER_H

#include "FastSimulation/Particle/interface/RawParticle.h"

/**
 * Abstract base class for filtering of RawParticle s.
 * Filters can be chained - with logical AND between them.
 * \author Stephan Wynhoff
 */

class BaseRawParticleFilter  {
public:

  BaseRawParticleFilter(){;};

  virtual ~BaseRawParticleFilter(){;};

public:

  bool accept(const RawParticle& p) const;

  bool accept(const RawParticle* p) const;

  /// Add a BaseRawParticleFilter to be run after executing this one.
  void addFilter(BaseRawParticleFilter* f);

protected:

  /// Here the specific filtering is to be done.
  virtual bool isOKForMe(const RawParticle* p) const = 0;

private:
  std::vector<BaseRawParticleFilter*> myFilter;
};

#endif
