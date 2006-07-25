#ifndef CandAlgos_SingleCandidateSelector_h
#define CandAlgosSingleCandidateSelector_h
/* \class SingleCandidateSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.1 2006/07/25 09:02:56 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include  <boost/shared_ptr.hpp>

struct SingleCandidateSelector {
  SingleCandidateSelector( const edm::ParameterSet & cfg );
  bool operator()( const reco::Candidate & ) const;
private:
  boost::shared_ptr<CandSelector> select_;
};

#endif
