#ifndef CandUtils_CandSelector_h
#define CandUtils_CandSelector_h
// $Id$

namespace aod {
  class Candidate;
}

class CandSelector {
public:
  virtual ~CandSelector();
  virtual bool operator()( const aod::Candidate & c ) const = 0;
};

#endif
