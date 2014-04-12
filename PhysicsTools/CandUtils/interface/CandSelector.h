#ifndef CandUtils_CandSelector_h
#define CandUtils_CandSelector_h
/** \class CandSelector
 *
 * Base class for all candidate selector 
 *
 * \author Luca Lista, INFN
 *
 *
 *
 */

namespace reco {
  class Candidate;
}

class CandSelector {
public:
  /// destructor
  virtual ~CandSelector();
  /// return true if the candidate is selected
  virtual bool operator()( const reco::Candidate & c ) const = 0;
};

#endif
