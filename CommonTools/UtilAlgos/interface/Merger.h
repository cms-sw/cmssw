#ifndef UtilAlgos_Merger_h
#define UtilAlgos_Merger_h
/** \class Merger
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
 * $Id: Merger.h,v 1.2 2010/02/20 20:55:21 wmtan Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <vector>

template<typename InputCollection,
	 typename OutputCollection = InputCollection,
	 typename P = typename edm::clonehelper::CloneTrait<InputCollection>::type>
class Merger : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit Merger( const edm::ParameterSet& );
  /// destructor
  ~Merger();

private:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup&) override;
  /// vector of strings
  typedef std::vector<edm::EDGetTokenT<InputCollection> > vtoken;
  /// labels of the collections to be merged
  vtoken srcToken_;
};

template<typename InputCollection, typename OutputCollection, typename P>
Merger<InputCollection, OutputCollection, P>::Merger( const edm::ParameterSet& par ) :
  srcToken_( edm::vector_transform(par.template getParameter<std::vector<edm::InputTag> >( "src" ), [this](edm::InputTag const & tag){return consumes<InputCollection>(tag);} ) ) {
  produces<OutputCollection>();
}

template<typename InputCollection, typename OutputCollection, typename P>
Merger<InputCollection, OutputCollection, P>::~Merger() {
}

template<typename InputCollection, typename OutputCollection, typename P>
void Merger<InputCollection, OutputCollection, P>::produce( edm::Event& evt, const edm::EventSetup&) {
  std::auto_ptr<OutputCollection> coll( new OutputCollection );
  for( typename vtoken::const_iterator s = srcToken_.begin(); s != srcToken_.end(); ++ s ) {
    edm::Handle<InputCollection> h;
    evt.getByToken( * s, h );
    for( typename InputCollection::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll->push_back( P::clone( * c ) );
    }
  }
  evt.put( coll );
}

#endif

