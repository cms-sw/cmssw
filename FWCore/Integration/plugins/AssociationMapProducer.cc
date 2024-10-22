/** \class edm::AssociationMapProducer
\author W. David Dagenhart, created 10 March 2015
*/

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
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
    ~AssociationMapProducer() override;

    void produce(edm::Event&, edm::EventSetup const&) override;

    typedef edm::AssociationMap<edm::OneToValue<std::vector<int>, double> > AssocOneToValue;
    typedef edm::AssociationMap<edm::OneToOne<std::vector<int>, std::vector<int> > > AssocOneToOne;
    typedef edm::AssociationMap<edm::OneToMany<std::vector<int>, std::vector<int> > > AssocOneToMany;
    typedef edm::AssociationMap<edm::OneToManyWithQuality<std::vector<int>, std::vector<int>, double> >
        AssocOneToManyWithQuality;
    typedef edm::AssociationMap<edm::OneToOne<edm::View<int>, edm::View<int> > > AssocOneToOneView;

  private:
    edm::EDGetTokenT<std::vector<int> > inputToken1_;
    edm::EDGetTokenT<std::vector<int> > inputToken2_;

    edm::EDGetTokenT<edm::View<int> > inputToken1V_;
    edm::EDGetTokenT<edm::View<int> > inputToken2V_;
  };

  AssociationMapProducer::AssociationMapProducer(edm::ParameterSet const& pset) {
    inputToken1_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag1"));
    inputToken2_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag2"));

    inputToken1V_ = consumes<edm::View<int> >(pset.getParameter<edm::InputTag>("inputTag1"));
    inputToken2V_ = consumes<edm::View<int> >(pset.getParameter<edm::InputTag>("inputTag2"));

    produces<AssocOneToOne>();
    produces<AssocOneToOne>("twoArg");
    produces<AssocOneToValue>();
    produces<AssocOneToValue>("handleArg");
    produces<AssocOneToMany>();
    produces<AssocOneToManyWithQuality>();
    produces<AssocOneToOneView>();
    produces<AssocOneToOneView>("twoArg");
  }

  AssociationMapProducer::~AssociationMapProducer() {}

  void AssociationMapProducer::produce(edm::Event& event, edm::EventSetup const&) {
    edm::Handle<std::vector<int> > inputCollection1 = event.getHandle(inputToken1_);

    edm::Handle<std::vector<int> > inputCollection2 = event.getHandle(inputToken2_);

    // insert some entries into some AssociationMaps, in another
    // module we will readout the contents and check that we readout
    // the same content as was put in. Note that the particular values
    // used are arbitrary and have no meaning.

    auto assoc1 = std::make_unique<AssocOneToOne>(&event.productGetter());
    assoc1->insert(edm::Ref<std::vector<int> >(inputCollection1, 0), edm::Ref<std::vector<int> >(inputCollection2, 1));
    assoc1->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), edm::Ref<std::vector<int> >(inputCollection2, 3));
    event.put(std::move(assoc1));

    auto assoc2 = std::make_unique<AssocOneToOne>(inputCollection1, inputCollection2);
    assoc2->insert(edm::Ref<std::vector<int> >(inputCollection1, 0), edm::Ref<std::vector<int> >(inputCollection2, 1));
    assoc2->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), edm::Ref<std::vector<int> >(inputCollection2, 4));
    event.put(std::move(assoc2), "twoArg");

    auto assoc3 = std::make_unique<AssocOneToValue>(&event.productGetter());
    assoc3->insert(edm::Ref<std::vector<int> >(inputCollection1, 0), 11.0);
    assoc3->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), 12.0);
    event.put(std::move(assoc3));

    auto assoc4 = std::make_unique<AssocOneToValue>(inputCollection1);
    assoc4->insert(edm::Ref<std::vector<int> >(inputCollection1, 0), 21.0);
    assoc4->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), 22.0);
    event.put(std::move(assoc4), "handleArg");

    auto assoc5 = std::make_unique<AssocOneToMany>(&event.productGetter());
    assoc5->insert(edm::Ref<std::vector<int> >(inputCollection1, 0), edm::Ref<std::vector<int> >(inputCollection2, 1));
    assoc5->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), edm::Ref<std::vector<int> >(inputCollection2, 4));
    assoc5->insert(edm::Ref<std::vector<int> >(inputCollection1, 2), edm::Ref<std::vector<int> >(inputCollection2, 6));
    event.put(std::move(assoc5));

    auto assoc6 = std::make_unique<AssocOneToManyWithQuality>(&event.productGetter());
    assoc6->insert(edm::Ref<std::vector<int> >(inputCollection1, 0),
                   AssocOneToManyWithQuality::data_type(edm::Ref<std::vector<int> >(inputCollection2, 1), 31.0));
    assoc6->insert(edm::Ref<std::vector<int> >(inputCollection1, 2),
                   AssocOneToManyWithQuality::data_type(edm::Ref<std::vector<int> >(inputCollection2, 4), 32.0));
    assoc6->insert(edm::Ref<std::vector<int> >(inputCollection1, 2),
                   AssocOneToManyWithQuality::data_type(edm::Ref<std::vector<int> >(inputCollection2, 7), 33.0));
    event.put(std::move(assoc6));

    edm::View<int> const& inputView1 = event.get(inputToken1V_);

    edm::Handle<edm::View<int> > inputView2 = event.getHandle(inputToken2V_);

    auto assoc7 = std::make_unique<AssocOneToOneView>(&event.productGetter());
    assoc7->insert(inputView1.refAt(0), inputView2->refAt(3));
    assoc7->insert(inputView1.refAt(2), inputView2->refAt(4));
    event.put(std::move(assoc7));

    auto assoc8 = std::make_unique<AssocOneToOneView>(edm::makeRefToBaseProdFrom(inputView1.refAt(0), event),
                                                      edm::makeRefToBaseProdFrom(inputView2->refAt(0), event));

    assoc8->insert(inputView1.refAt(0), inputView2->refAt(5));
    assoc8->insert(inputView1.refAt(2), inputView2->refAt(6));
    event.put(std::move(assoc8), "twoArg");
  }
}  // namespace edmtest
using edmtest::AssociationMapProducer;
DEFINE_FWK_MODULE(AssociationMapProducer);
