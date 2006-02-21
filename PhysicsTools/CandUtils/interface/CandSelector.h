#ifndef CandUtils_CandSelector_h
#define CandUtils_CandSelector_h
// $Id: CandSelector.h,v 1.1 2005/10/25 09:08:31 llista Exp $

namespace reco {
  class Candidate;
}

class CandSelector {
public:
  virtual ~CandSelector();
  virtual bool operator()( const reco::Candidate & c ) const = 0;
};

#endif
