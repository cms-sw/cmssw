#ifndef PhysicsTools_UtilAlgos_CollectionAdder_h
#define PhysicsTools_UtilAlgos_CollectionAdder_h
/* \class CollectionAdder<C>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CollectionAdder.h,v 1.3 2010/02/20 20:55:17 wmtan Exp $
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C>
class CollectionAdder : public edm::EDProducer {
public:
  typedef C collection;
  CollectionAdder(const edm::ParameterSet & cfg ) :
    srcTokens_(edm::vector_transform(cfg.template getParameter<std::vector<edm::InputTag> >("src"), [this](edm::InputTag const & tag){return consumes<collection>(tag);})) {
    produces<collection>();
  }
private:
  std::vector<edm::EDGetTokenT<collection>> srcTokens_;
  void produce(edm::Event & evt, const edm::EventSetup&) override {
    std::auto_ptr<collection> coll(new collection);
    typename collection::Filler filler(*coll);
    for(size_t i = 0; i < srcTokens_.size(); ++i ) {
      edm::Handle<collection> src;
      evt.getByToken(srcTokens_[i], src);
      *coll += *src;
    }
    evt.put(coll);
  }
};

#endif

