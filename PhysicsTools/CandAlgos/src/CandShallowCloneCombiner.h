#ifndef CandAlgos_CandShallowCloneCombiner_h
#define CandAlgos_CandShallowCloneCombiner_h
/** \class CandShallowCloneCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the
 * NBodyShallowCloneCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.10 $
 *
 * $Id: CandShallowCloneCombiner.h,v 1.10 2006/10/11 10:08:59 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/NBodyShallowCloneCombiner.h"
#include "PhysicsTools/CandAlgos/src/decayParser.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
}

class CandShallowCloneCombinerBase : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit CandShallowCloneCombinerBase( const edm::ParameterSet & );    
  /// destructor
  virtual ~CandShallowCloneCombinerBase();
  
protected:
  /// daughter charges
  std::vector<int> dauCharge_;
  /// label vector
  std::vector<cand::parser::ConjInfo> labels_;
};

template<typename S>
class CandShallowCloneCombiner : public CandShallowCloneCombinerBase {
public:
  /// constructor from parameter set
  explicit CandShallowCloneCombiner( const edm::ParameterSet & cfg ) :
    CandShallowCloneCombinerBase( cfg ), combiner_( cfg, true, dauCharge_ ) {
  }
  /// destructor
  virtual ~CandShallowCloneCombiner() { }
private:
  /// process an event
  void produce( edm::Event& evt, const edm::EventSetup& ) {
    int n = labels_.size();
    std::vector<edm::Handle<reco::CandidateCollection> > colls( n );
    for( int i = 0; i < n; ++i )
      evt.getByLabel( labels_[ i ].tag_, colls[ i ] );
    
    std::vector<reco::CandidateRefProd> cv;
    for( std::vector<edm::Handle<reco::CandidateCollection> >::const_iterator c = colls.begin();
	 c != colls.end(); ++ c )
      cv.push_back( reco::CandidateRefProd( * c ) );
    
    evt.put( combiner_.combine( cv ) );
  }
  /// combiner utility
  NBodyShallowCloneCombiner<S> combiner_;
};

#endif
