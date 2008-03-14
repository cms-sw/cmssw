#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the CandCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.6 $
 *
 * $Id: CandCombiner.h,v 1.6 2007/10/31 14:12:35 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/CandCombiner.h"
#include "PhysicsTools/CandAlgos/interface/decayParser.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "PhysicsTools/Utilities/interface/cutParser.h"
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace reco {
  namespace modules {

    struct CandCombinerBase : public edm::EDProducer {
      CandCombinerBase( const edm::ParameterSet & cfg ) {
        using namespace cand::parser;
	std::string decay(cfg.getParameter<std::string>("decay"));
	if( decayParser(decay, labels_))
	  for( std::vector<ConjInfo>::iterator label = labels_.begin();
	       label != labels_.end(); ++label )
	if( label->mode_ == ConjInfo::kPlus )
	  dauCharge_.push_back( 1 );
	else if ( label->mode_ == ConjInfo::kMinus )
	  dauCharge_.push_back( -1 );
	else
	  dauCharge_.push_back( 0 );
	else
	  throw edm::Exception( edm::errors::Configuration,
				"failed to parse \"" + decay + "\"" );
	
	int lists = labels_.size();
	if ( lists != 2 && lists != 3 ) {
	  throw edm::Exception( edm::errors::LogicError,
				"invalid number of collections" );
	}
      }
      /// label vector
      std::vector<cand::parser::ConjInfo> labels_;
      /// daughter charges
      std::vector<int> dauCharge_;
    };
    
    template<typename InputCollection, 
	     typename Selector, 
             typename OutputCollection = typename combiner::helpers::CandRefHelper<InputCollection>::OutputCollection,
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
		   checkCharge(cfg), 
		   dauCharge_ ) {
        produces<OutputCollection>();
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
	std::auto_ptr<OutputCollection> out = combiner_.combine(cv);
	evt.put(out);
      }
      /// combiner utility
      ::CandCombiner<InputCollection, Selector, OutputCollection, PairSelector, Cloner, Setup> combiner_;
      bool checkCharge( const edm::ParameterSet & cfg ) const {
	using namespace std;
	const string par( "checkCharge" );
	vector<string> bools = cfg.getParameterNamesForType<bool>();
	bool found = find( bools.begin(), bools.end(), "checkCharge" ) != bools.end();
	if (found) return cfg.getParameter<bool>( par );
	// default: check charge
	return true;
      }
    };

  }
}

#endif
