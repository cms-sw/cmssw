#ifndef UtilAlgos_SetMerger_h
#define UtilAlgos_SetMerger_h
/** \class SetMerger
 *
 * Merges an arbitrary number of collections
 * into a single collection.
 *
 * Template parameters:
 * - C : collection type
 * - P : policy class that specifies how objects
 *       in the collection are are cloned
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SetMerger.h,v 1.2 2010/02/20 20:55:21 wmtan Exp $
 *
 */
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <vector>

template<typename InputCollection,
	 typename OutputCollection = InputCollection,
	 typename P = typename edm::clonehelper::CloneTrait<InputCollection>::type>
class SetMerger : public edm::global::EDProducer<> {
public:
typedef std::set<typename OutputCollection::value_type > set_type;
  /// constructor from parameter set
  explicit SetMerger( const edm::ParameterSet& );
  /// destructor
  ~SetMerger() override;

private:
  /// process an event
  void produce( edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  /// vector of strings
  typedef std::vector<edm::EDGetTokenT<InputCollection> > vtoken;
  /// labels of the collections to be merged
  vtoken srcToken_;
};

template<typename InputCollection, typename OutputCollection, typename P>
SetMerger<InputCollection, OutputCollection, P>::SetMerger( const edm::ParameterSet& par ) :
  srcToken_( edm::vector_transform(par.template getParameter<std::vector<edm::InputTag> >( "src" ), [this](edm::InputTag const & tag){return consumes<InputCollection>(tag);} ) ) {
  produces<OutputCollection>();
}

template<typename InputCollection, typename OutputCollection, typename P>
SetMerger<InputCollection, OutputCollection, P>::~SetMerger() {
}

template<typename InputCollection, typename OutputCollection, typename P>
void SetMerger<InputCollection, OutputCollection, P>::produce( edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
  set_type coll_set;
  for( typename vtoken::const_iterator s = srcToken_.begin(); s != srcToken_.end(); ++ s ) {
    edm::Handle<InputCollection> h;
    evt.getByToken( * s, h );
    for( typename InputCollection::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll_set.emplace( P::clone( * c ) );
    }
  }
  std::unique_ptr<OutputCollection> coll( new OutputCollection( coll_set.size() ) );
  std::copy(coll_set.begin(), coll_set.end(), coll->begin());
  evt.put(std::move(coll));
}

#endif

