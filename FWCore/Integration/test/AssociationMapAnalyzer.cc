/** \class edm::AssociationMapAnalyzer
\author W. David Dagenhart, created 10 March 2015
*/

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

namespace edm {
  class EventSetup;
  class StreamID;
}

namespace edmtest {

  class AssociationMapAnalyzer : public edm::global::EDAnalyzer<> {
  public:

    typedef edm::AssociationMap<edm::OneToValue<std::vector<int>, double > > AssocOneToValue;
    typedef edm::AssociationMap<edm::OneToOne<std::vector<int>, std::vector<int> > > AssocOneToOne;
    typedef edm::AssociationMap<edm::OneToMany<std::vector<int>, std::vector<int> > > AssocOneToMany;
    typedef edm::AssociationMap<edm::OneToManyWithQuality<std::vector<int>, std::vector<int>, double > > AssocOneToManyWithQuality;
    typedef edm::AssociationMap<edm::OneToOne<edm::View<int>, edm::View<int> > > AssocOneToOneView;

    explicit AssociationMapAnalyzer(edm::ParameterSet const&);
    virtual void analyze(edm::StreamID,
                         edm::Event const& event,
                         edm::EventSetup const&) const override;

    edm::EDGetTokenT<std::vector<int> > inputToken1_;
    edm::EDGetTokenT<std::vector<int> > inputToken2_;
    edm::EDGetTokenT<edm::View<int> > inputToken1V_;
    edm::EDGetTokenT<edm::View<int> > inputToken2V_;

    edm::EDGetTokenT<AssocOneToOne> associationMapToken1_;
    edm::EDGetTokenT<AssocOneToOne> associationMapToken2_;
    edm::EDGetTokenT<AssocOneToValue> associationMapToken3_;
    edm::EDGetTokenT<AssocOneToValue> associationMapToken4_;
    edm::EDGetTokenT<AssocOneToMany> associationMapToken5_;
    edm::EDGetTokenT<AssocOneToManyWithQuality> associationMapToken6_;
    edm::EDGetTokenT<AssocOneToOneView> associationMapToken7_;
    edm::EDGetTokenT<AssocOneToOneView> associationMapToken8_;
  };


  AssociationMapAnalyzer::AssociationMapAnalyzer(edm::ParameterSet const& pset) {
    inputToken1_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag1"));
    inputToken2_ = consumes<std::vector<int> >(pset.getParameter<edm::InputTag>("inputTag2"));
    inputToken1V_ = consumes<edm::View<int> >(pset.getParameter<edm::InputTag>("inputTag1"));
    inputToken2V_ = consumes<edm::View<int> >(pset.getParameter<edm::InputTag>("inputTag2"));
    associationMapToken1_ = consumes<AssocOneToOne>(pset.getParameter<edm::InputTag>("associationMapTag1"));
    associationMapToken2_ = consumes<AssocOneToOne>(pset.getParameter<edm::InputTag>("associationMapTag2"));
    associationMapToken3_ = consumes<AssocOneToValue>(pset.getParameter<edm::InputTag>("associationMapTag3"));
    associationMapToken4_ = consumes<AssocOneToValue>(pset.getParameter<edm::InputTag>("associationMapTag4"));
    associationMapToken5_ = consumes<AssocOneToMany>(pset.getParameter<edm::InputTag>("associationMapTag5"));
    associationMapToken6_ = consumes<AssocOneToManyWithQuality>(pset.getParameter<edm::InputTag>("associationMapTag6"));
    associationMapToken7_ = consumes<AssocOneToOneView>(pset.getParameter<edm::InputTag>("associationMapTag7"));
    associationMapToken8_ = consumes<AssocOneToOneView>(pset.getParameter<edm::InputTag>("associationMapTag8"));
  }

  void
  AssociationMapAnalyzer::analyze(edm::StreamID,
                                  edm::Event const& event,
                                  edm::EventSetup const&) const {

    edm::Handle<std::vector<int> > inputCollection1;
    event.getByToken(inputToken1_, inputCollection1);

    edm::Handle<std::vector<int> > inputCollection2;
    event.getByToken(inputToken2_, inputCollection2);

    // Readout some entries from some AssociationMaps and check that
    // we readout the same content as was was put in. We know the values
    // by looking at the hard coded values in AssociationMapProducer.
    // The particular values used are arbitrary and have no meaning.

    edm::Handle<AssocOneToOne> hAssociationMap1;
    event.getByToken(associationMapToken1_, hAssociationMap1);
    AssocOneToOne const& associationMap1 = *hAssociationMap1;

    if(*associationMap1[edm::Ref<std::vector<int> >(inputCollection1, 0)] != 22 ||
       *associationMap1[edm::Ref<std::vector<int> >(inputCollection1, 2)] != 24 ||
       *associationMap1[edm::Ptr<int>(inputCollection1, 0)] != 22 ||
       *associationMap1[edm::Ptr<int>(inputCollection1, 2)] != 24) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 1";
    }
    AssocOneToOne::const_iterator iter = associationMap1.begin();
    if(*iter->val != 22 || iter->key != edm::Ref<std::vector<int> >(inputCollection1, 0)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 2";
    }
    ++iter;
    if(*iter->val != 24 || iter->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 3";
    }
    ++iter;
    if(iter != associationMap1.end()) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 4";
    }

    // Case where handle arguments were used creating the AssociationMap

    edm::Handle<AssocOneToOne> hAssociationMap2;
    event.getByToken(associationMapToken2_, hAssociationMap2);
    AssocOneToOne const& associationMap2 = *hAssociationMap2;

    if(*associationMap2[edm::Ref<std::vector<int> >(inputCollection1, 0)] != 22 ||
       *associationMap2[edm::Ref<std::vector<int> >(inputCollection1, 2)] != 25 ||
       *associationMap2[edm::Ptr<int>(inputCollection1, 0)] != 22 ||
       *associationMap2[edm::Ptr<int>(inputCollection1, 2)] != 25) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 5";
    }

    AssocOneToOne::const_iterator iter2 = associationMap2.begin();
    ++iter2;
    if(*iter2->val != 25 || iter2->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 6";
    }

    // One to Value case

    edm::Handle<AssocOneToValue> hAssociationMap3;
    event.getByToken(associationMapToken3_, hAssociationMap3);
    AssocOneToValue const& associationMap3 = *hAssociationMap3;

    if(associationMap3[edm::Ref<std::vector<int> >(inputCollection1, 0)] != 11.0 ||
       associationMap3[edm::Ref<std::vector<int> >(inputCollection1, 2)] != 12.0 ||
       associationMap3[edm::Ptr<int>(inputCollection1, 0)] != 11.0 ||
       associationMap3[edm::Ptr<int>(inputCollection1, 2)] != 12.0) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 7";
    }
    AssocOneToValue::const_iterator iter3 = associationMap3.begin();
    ++iter3;
    if(iter3->val != 12.0 || iter3->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 8";
    }

    // One to Value case with handle argument

    edm::Handle<AssocOneToValue> hAssociationMap4;
    event.getByToken(associationMapToken4_, hAssociationMap4);
    AssocOneToValue const& associationMap4 = *hAssociationMap4;

    if(associationMap4[edm::Ref<std::vector<int> >(inputCollection1, 0)] != 21.0 ||
       associationMap4[edm::Ref<std::vector<int> >(inputCollection1, 2)] != 22.0 ||
       associationMap4[edm::Ptr<int>(inputCollection1, 0)] != 21.0 ||
       associationMap4[edm::Ptr<int>(inputCollection1, 2)] != 22.0) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 9";
    }
    AssocOneToValue::const_iterator iter4 = associationMap4.begin();
    ++iter4;
    if(iter4->val != 22.0 || iter4->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 10";
    }

    // One to Many

    edm::Handle<AssocOneToMany> hAssociationMap5;
    event.getByToken(associationMapToken5_, hAssociationMap5);
    AssocOneToMany const& associationMap5 = *hAssociationMap5;

    if(*associationMap5[edm::Ref<std::vector<int> >(inputCollection1, 0)].at(0) != 22 ||
       *associationMap5[edm::Ref<std::vector<int> >(inputCollection1, 2)].at(1) != 27 ||
       *associationMap5[edm::Ptr<int>(inputCollection1, 0)].at(0) != 22 ||
       *associationMap5[edm::Ptr<int>(inputCollection1, 2)].at(1) != 27) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 11";
    }
    AssocOneToMany::const_iterator iter5 = associationMap5.begin();
    ++iter5;
    if(*iter5->val.at(1) != 27 || iter5->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 12";
    }

    // One to Many With Quality

    edm::Handle<AssocOneToManyWithQuality> hAssociationMap6;
    event.getByToken(associationMapToken6_, hAssociationMap6);
    AssocOneToManyWithQuality const& associationMap6 = *hAssociationMap6;
    if(*associationMap6[edm::Ref<std::vector<int> >(inputCollection1, 0)].at(0).first != 22 ||
       *associationMap6[edm::Ref<std::vector<int> >(inputCollection1, 2)].at(1).first != 25 ||
       *associationMap6[edm::Ptr<int>(inputCollection1, 0)].at(0).first != 22 ||
       *associationMap6[edm::Ptr<int>(inputCollection1, 2)].at(1).first != 25 ||
       associationMap6[edm::Ref<std::vector<int> >(inputCollection1, 0)].at(0).second != 31.0 ||
       associationMap6[edm::Ref<std::vector<int> >(inputCollection1, 2)].at(1).second != 32.0 ||
       associationMap6[edm::Ptr<int>(inputCollection1, 0)].at(0).second != 31.0 ||
       associationMap6[edm::Ptr<int>(inputCollection1, 2)].at(1).second != 32.0) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 13";
    }
    AssocOneToManyWithQuality::const_iterator iter6 = associationMap6.begin();
    ++iter6;
    if(*iter6->val.at(1).first != 25 ||
       iter6->val.at(1).second != 32.0 ||
       iter6->key != edm::Ref<std::vector<int> >(inputCollection1, 2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 14";
    }

    // One to One View

    edm::Handle<edm::View<int> > inputView1;
    event.getByToken(inputToken1V_, inputView1);

    edm::Handle<edm::View<int> > inputView2;
    event.getByToken(inputToken2V_, inputView2);

    edm::Handle<AssocOneToOneView> hAssociationMap7;
    event.getByToken(associationMapToken7_, hAssociationMap7);
    AssocOneToOneView const& associationMap7 = *hAssociationMap7;
    if(*associationMap7[inputView1->refAt(0)] != 24 ||
       *associationMap7[inputView1->refAt(2)] != 25) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 15";
    }
    AssocOneToOneView::const_iterator iter7 = associationMap7.begin();
    ++iter7;
    if(*iter7->val != 25 ||
       iter7->key != inputView1->refAt(2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 16";
    }

    // One to One View built with 2 arguments constructor

    edm::Handle<AssocOneToOneView> hAssociationMap8;
    event.getByToken(associationMapToken8_, hAssociationMap8);
    AssocOneToOneView const& associationMap8 = *hAssociationMap8;
    if(*associationMap8[inputView1->refAt(0)] != 26 ||
       *associationMap8[inputView1->refAt(2)] != 27) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 17";
    }
    AssocOneToOneView::const_iterator iter8 = associationMap8.begin();
    ++iter8;
    if(*iter8->val != 27 ||
       iter8->key != inputView1->refAt(2)) {
      throw cms::Exception("TestFailure") << "unexpected result after using AssociationMap 18";
    }
  }
}
using edmtest::AssociationMapAnalyzer;
DEFINE_FWK_MODULE(AssociationMapAnalyzer);
