#ifndef CandAlgos_CandCombiner_h
#define CandAlgos_CandCombiner_h
/** \class CandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the CandCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: CandCombiner.h,v 1.3 2007/06/17 09:20:54 llista Exp $
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
#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace combiner {
  namespace helpers {
    template<typename Setup>
      struct SetupInit {
	static void init( Setup & s, const edm::EventSetup& es ) { }
      };
  }
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
    
    template<typename S, typename H = ::combiner::helpers::NormalClone, typename Setup = AddFourMomenta,
      typename Init = ::combiner::helpers::SetupInit<Setup> >
    class CandCombiner : public CandCombinerBase {
    public:
      /// constructor from parameter set
      explicit CandCombiner( const edm::ParameterSet & cfg ) :
      CandCombinerBase( cfg ), 
      combiner_( reco::modules::make<S>( cfg ), 
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
    ::CandCombiner<S, H, Setup> combiner_;
    };

  }
}

#endif
