/** \class edm::AssociationMapProducer
\author W. David Dagenhart, created 10 March 2015
*/

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class AssociationMapProducer : public edm::one::EDProducer<> {
  public:

    explicit AssociationMapProducer(edm::ParameterSet const&);
    virtual ~AssociationMapProducer();

    void produce(edm::Event&, edm::EventSetup const&) override;

    typedef edm::AssociationMap<edm::OneToOne<std::vector<int>, std::vector<int> > > AssocOneToOne;

  private:

    edm::EDGetTokenT<std::vector<int> > inputToken_;
  };

  AssociationMapProducer::AssociationMapProducer(edm::ParameterSet const& pset) {

    inputToken_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag"));

    produces<AssocOneToOne>();
  }

  AssociationMapProducer::~AssociationMapProducer() { }

  void AssociationMapProducer::produce(edm::Event& event, edm::EventSetup const&) {

    edm::Handle<std::vector<int> > inputCollection;
    event.getByToken(inputToken_, inputCollection);
    std::vector<int> vint = *inputCollection;

    std::auto_ptr<AssocOneToOne> result(new AssocOneToOne(&event.productGetter()));
    result->insert(edm::Ref<std::vector<int> >(inputCollection, 0),
                  edm::Ref<std::vector<int> >(inputCollection, 1));
    result->insert(edm::Ref<std::vector<int> >(inputCollection, 2),
                  edm::Ref<std::vector<int> >(inputCollection, 3));
    event.put(result);
  }
}
using edmtest::AssociationMapProducer;
DEFINE_FWK_MODULE(AssociationMapProducer);
