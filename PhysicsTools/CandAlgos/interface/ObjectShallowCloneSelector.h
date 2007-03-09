#ifndef CandAlgos_ObjectShallowCloneSelector_h
#define CandAlgos_ObjectShallowCloneSelector_h
/* \class RefVectorShallowCloneStoreMananger
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace helper {
  
  struct RefVectorShallowCloneStoreMananger {
    typedef reco::CandidateCollection collection;
    RefVectorShallowCloneStoreMananger() : selected_( new reco::CandidateCollection ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i )
	selected_->push_back( new reco::ShallowCloneCandidate( reco::CandidateBaseRef( * i ) ) );
    }
    edm::OrphanHandle<collection> put( edm::Event & evt ) {
      return evt.put( selected_ );
    }
    size_t size() const { return selected_->size(); }
  private:
    std::auto_ptr<reco::CandidateCollection> selected_;
  };

}

template<typename Selector, 
	 typename SizeSelector = NonNullNumberSelector,
         typename PostProcessor = reco::helpers::NullPostProcessor<reco::CandidateCollection> >
class ObjectShallowCloneSelector : public ObjectSelector<Selector, reco::CandidateCollection, SizeSelector, 
							 PostProcessor, helper::RefVectorShallowCloneStoreMananger> {
public:
  explicit ObjectShallowCloneSelector( const edm::ParameterSet & cfg ) :
    ObjectSelector<Selector, reco::CandidateCollection, SizeSelector, 
		   PostProcessor, helper::RefVectorShallowCloneStoreMananger>( cfg ) { }
};

#endif
