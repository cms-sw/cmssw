#ifndef PHYSICSTOOLS_SELECTOR_H
#define PHYSICSTOOLS_SELECTOR_H
// $Id: Selector.h,v 1.1 2005/10/21 12:44:35 llista Exp $

namespace aod {
  class Candidate;

  class Selector {
  public:
    virtual ~Selector();
    virtual bool operator()( const aod::Candidate & c ) const = 0;
  };
}

#endif
