#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the CandCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.9 $
 *
 * $Id: CandCombiner.h,v 1.9 2007/07/23 10:04:07 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/CandCombiner.h"
#include "PhysicsTools/CandAlgos/src/decayParser.h"
#include "PhysicsTools/Parser/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace reco {
  namespace modules {
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
    
    template<typename InputCollection, 
	     typename Selector, 
             typename PairSelector = AnyPairSelector,
             typename Cloner = ::combiner::helpers::NormalClone, 
             typename Setup = AddFourMomenta,
             typename Init = typename ::reco::modules::EventSetupInit<Setup>::type
            >
    class CandCombiner : public CandCombinerBase {
    public:
      /// constructor from parameter settypedef 
      explicit CandCombiner( const edm::ParameterSet & cfg ) :
      CandCombinerBase( cfg ), 
      combiner_( reco::modules::make<Selector>( cfg ), 
		 reco::modules::make<PairSelector>( cfg ),
		 Setup( cfg ), 
		 true, 
		 dauCharge_ ) {
		 }
      /// destructor
      virtual ~CandCombiner() { }

    private:
    /// process an event
    void produce( edm::Event& evt, const edm::EventSetup& es ) {
      Init::init( combiner_.setup(), es );
      int n = labels_.size();
      std::vector<edm::Handle<InputCollection> > colls( n );
      for( int i = 0; i < n; ++i )
      evt.getByLabel( labels_[ i ].tag_, colls[ i ] );
      typedef typename combiner::helpers::template CandRefHelper<InputCollection>::RefProd RefProd;
      std::vector<RefProd> cv;
      for( typename std::vector<edm::Handle<InputCollection> >::const_iterator c = colls.begin();
	   c != colls.end(); ++ c ) {
	RefProd r( *c );
	cv.push_back( r );
      }
      
      evt.put( combiner_.combine( cv ) );
    }
    
    /// combiner utility
    ::CandCombiner<InputCollection, Selector, PairSelector, Cloner, Setup> combiner_;
    };

  }
}

#endif
