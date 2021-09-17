// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      ProducerUsingCollector
//
/**\class edmtest::ProducerUsingCollector

  Description: Used in tests of the ProducesCollector. It uses all
the different functions in ProducesCollector for no reason other
than to test them all.
*/
// Original Author:  W. David Dagenhart
//         Created:  26 September 2019

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edmtest {

  class ProducerHelperUsingCollector {
  public:
    ProducerHelperUsingCollector(edm::ProducesCollector&&);
    void putEventProducts(edm::Event&) const;
    void putBeginRunProducts(edm::Run&) const;
    void putEndRunProducts(edm::Run&) const;
    void putBeginLumiProducts(edm::LuminosityBlock&) const;
    void putEndLumiProducts(edm::LuminosityBlock&) const;

  private:
    edm::EDPutTokenT<IntProduct> eventToken_;
    edm::EDPutTokenT<IntProduct> eventWithInstanceToken_;
    edm::EDPutTokenT<UInt64Product> eventWithTransitionToken_;
    edm::EDPutToken eventUsingTypeIDToken_;
    edm::EDPutTokenT<IntProduct> brToken_;
    edm::EDPutTokenT<IntProduct> erToken_;
    edm::EDPutToken blToken_;
    edm::EDPutToken elToken_;
  };

  ProducerHelperUsingCollector::ProducerHelperUsingCollector(edm::ProducesCollector&& producesCollector)
      : eventToken_(producesCollector.produces<IntProduct>()),
        eventWithInstanceToken_(producesCollector.produces<IntProduct>("event")),
        eventWithTransitionToken_(producesCollector.produces<UInt64Product, edm::Transition::Event>()),
        eventUsingTypeIDToken_(producesCollector.produces(edm::TypeID(typeid(IntProduct)), "eventOther")),
        brToken_(producesCollector.produces<IntProduct, edm::Transition::BeginRun>("beginRun")),
        blToken_(producesCollector.produces<edm::Transition::BeginLuminosityBlock>(edm::TypeID(typeid(IntProduct)),
                                                                                   "beginLumi")) {
    edm::ProducesCollector copy(producesCollector);
    erToken_ = copy.produces<IntProduct, edm::Transition::EndRun>("endRun");

    copy = producesCollector;
    edm::ProducesCollector copy2(producesCollector);
    copy2 = std::move(copy);
    elToken_ = copy.produces<edm::Transition::EndLuminosityBlock>(edm::TypeID(typeid(IntProduct)), "endLumi");
  }

  void ProducerHelperUsingCollector::putEventProducts(edm::Event& event) const {
    event.emplace(eventToken_, 1);
    event.emplace(eventWithInstanceToken_, 2);
    event.emplace(eventWithTransitionToken_, 3);
    event.put(eventUsingTypeIDToken_, std::make_unique<IntProduct>(4));
  }

  void ProducerHelperUsingCollector::putBeginRunProducts(edm::Run& run) const { run.emplace(brToken_, 5); }

  void ProducerHelperUsingCollector::putEndRunProducts(edm::Run& run) const { run.emplace(erToken_, 6); }

  void ProducerHelperUsingCollector::putBeginLumiProducts(edm::LuminosityBlock& luminosityBlock) const {
    luminosityBlock.put(blToken_, std::make_unique<IntProduct>(7));
  }

  void ProducerHelperUsingCollector::putEndLumiProducts(edm::LuminosityBlock& luminosityBlock) const {
    luminosityBlock.put(elToken_, std::make_unique<IntProduct>(8));
  }

  class ProducerUsingCollector : public edm::global::EDProducer<edm::BeginRunProducer,
                                                                edm::EndRunProducer,
                                                                edm::EndLuminosityBlockProducer,
                                                                edm::BeginLuminosityBlockProducer> {
  public:
    explicit ProducerUsingCollector(edm::ParameterSet const&);

    ~ProducerUsingCollector() override;

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override;

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override;

    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override;

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    ProducerHelperUsingCollector helper_;
  };

  ProducerUsingCollector::ProducerUsingCollector(edm::ParameterSet const&) : helper_(producesCollector()) {}

  ProducerUsingCollector::~ProducerUsingCollector() {}

  void ProducerUsingCollector::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
    helper_.putEventProducts(event);
  }

  void ProducerUsingCollector::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& lb,
                                                                 edm::EventSetup const&) const {
    helper_.putBeginLumiProducts(lb);
  }

  void ProducerUsingCollector::globalEndLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const&) const {
    helper_.putEndLumiProducts(lb);
  }

  void ProducerUsingCollector::globalBeginRunProduce(edm::Run& run, edm::EventSetup const&) const {
    helper_.putBeginRunProducts(run);
  }

  void ProducerUsingCollector::globalEndRunProduce(edm::Run& run, edm::EventSetup const&) const {
    helper_.putEndRunProducts(run);
  }

  void ProducerUsingCollector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using edmtest::ProducerUsingCollector;
DEFINE_FWK_MODULE(ProducerUsingCollector);
