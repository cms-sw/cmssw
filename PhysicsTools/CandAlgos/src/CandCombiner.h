#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the NBodyCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.11 $
 *
 * $Id: CandCombiner.h,v 1.11 2006/11/29 14:00:42 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/NBodyCombiner.h"
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

class CandCombinerBase : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit CandCombinerBase( const edm::ParameterSet & );    
  /// destructor
  virtual ~CandCombinerBase();
  
protected:
  /// daughter charges
  std::vector<int> dauCharge_;
  /// label vector
  std::vector<cand::parser::ConjInfo> labels_;
};

template<typename S, typename H = combiner::helpers::NormalClone, typename Setup = AddFourMomenta>
class CandCombiner : public CandCombinerBase {
public:
  /// constructor from parameter set
  explicit CandCombiner( const edm::ParameterSet & cfg ) :
    CandCombinerBase( cfg ), combiner_( cfg, true, dauCharge_ ) {
  }
  /// destructor
  virtual ~CandCombiner() { }
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
  NBodyCombiner<S, H, Setup> combiner_;
};

#endif
