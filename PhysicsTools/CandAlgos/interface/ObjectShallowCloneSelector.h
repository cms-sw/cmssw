#ifndef CandAlgos_ObjectShallowCloneSelector_h
#define CandAlgos_ObjectShallowCloneSelector_h
/* \class RefVectorShallowCloneStoreMananger
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementRefVectorCollectionSelector.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace helper {
  
  struct RefVectorShallowCloneStoreMananger {
    RefVectorShallowCloneStoreMananger() : selected_( new reco::CandidateCollection ) { 
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & ) {
      for( I i = begin; i != end; ++ i )
	selected_->push_back( new reco::ShallowCloneCandidate( reco::CandidateBaseRef( * i ) ) );
    }
    void put( edm::Event & evt ) {
      evt.put( selected_ );
    }
    bool empty() const { return selected_->empty(); }
  private:
    std::auto_ptr<reco::CandidateCollection> selected_;
  };

}

template<typename C, 
	 typename S, 
	 typename B = ObjectSelector<
                        SingleElementRefVectorCollectionSelector<C, S>,
                        helper::RefVectorShallowCloneStoreMananger,
                        helper::ObjectSelectorBase<reco::CandidateCollection>
                      > 
        >
class ObjectShallowCloneSelector : public B {
public:
  explicit ObjectShallowCloneSelector( const edm::ParameterSet & cfg ) :
    B( cfg ) { }
};

#endif
