#ifndef PHYSICSTOOLS_SELECTOR_H
#define PHYSICSTOOLS_SELECTOR_H
// $Id: PtMinSelector.h,v 1.3 2005/10/03 10:12:11 llista Exp $

namespace aod {
  class Candidate;

  class Selector {
  public:
    virtual ~Selector();
    virtual bool operator()( const aod::Candidate * c ) const = 0;
  };
}

#endif
