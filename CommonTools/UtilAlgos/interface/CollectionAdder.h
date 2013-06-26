#ifndef PhysicsTools_UtilAlgos_CollectionAdder_h
#define PhysicsTools_UtilAlgos_CollectionAdder_h
/* \class CollectionAdder<C>
 * 
 * \author Luca Lista, INFN
 * 
 * \version $Id: CollectionAdder.h,v 1.4 2013/02/28 00:34:12 wmtan Exp $
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C>
class CollectionAdder : public edm::EDProducer {
public:
  typedef C collection;
  CollectionAdder(const edm::ParameterSet & cfg ) :
    src_(cfg.template getParameter<std::vector<edm::InputTag> >("src")) {
    produces<collection>();
  }
private:
  std::vector<edm::InputTag> src_;
  void produce(edm::Event & evt, const edm::EventSetup&) override {
    std::auto_ptr<collection> coll(new collection);
    typename collection::Filler filler(*coll);
    for(size_t i = 0; i < src_.size(); ++i ) {
      edm::Handle<collection> src;
      evt.getByLabel(src_[i], src);
      *coll += *src;
    }
    evt.put(coll);
  }
};

#endif

