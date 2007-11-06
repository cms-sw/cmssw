#ifndef PhysicsTools_UtilAlgos_AssociationMerger_h
#define PhysicsTools_UtilAlgos_AssociationMerger_h
/* \class AssociationMerger<C>
 * 
 * \author Luca Lista, INFN
 * 
 * \version $Id$
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

template<typename C>
class AssociationMerger : public edm::EDProducer {
public:
  typedef edm::Association<C> collection;
  AssociationMerger(const edm::ParameterSet & cfg ) :
    src_(cfg.template getParameter<std::vector<edm::InputTag> >("src")) {
    produces<collection>();
  }
private:
  std::vector<edm::InputTag> src_;
  void produce(edm::Event & evt, const edm::EventSetup&) {
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
