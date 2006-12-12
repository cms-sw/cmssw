#ifndef HepMCCandAlgos_MCCandMatcher_h
#define HepMCCandAlgos_MCCandMatcher_h
/* \class MCCandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/CandUtils/interface/CandMatcher.h"

class MCCandMatcher : public CandMatcherBase {
public:
  /// constructor
  explicit MCCandMatcher( const reco::CandMatchMap & map );
  /// destructor
  virtual ~MCCandMatcher();
private:
  /// get ultimate daughter skipping status = 3
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const;
};

#endif
