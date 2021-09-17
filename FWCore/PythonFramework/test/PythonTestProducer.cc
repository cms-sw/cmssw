#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include <pybind11/pybind11.h>

#include <iostream>

namespace edmtest {
  class PythonTestProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  public:
    PythonTestProducer(edm::ParameterSet const&);

    void produce(edm::Event& iEvent, edm::EventSetup const&) override;

  private:
    edm::EDGetTokenT<IntProduct> get_;
    edm::EDPutTokenT<int> put_;
    int value_;
    pybind11::list outputList_;
  };

  PythonTestProducer::PythonTestProducer(edm::ParameterSet const& iPS)
      : get_(consumes<IntProduct>(iPS.getParameter<edm::InputTag>("source"))) {
    pybind11::module main_module = pybind11::module::import("__main__");
    auto main_namespace = main_module.attr("__dict__");

    //NOTE attempts to hold the object directly and read it in `produce` lead to segmentation faults
    value_ = main_namespace[(iPS.getParameter<std::string>("inputVariable")).c_str()].cast<int>();
    outputList_ = main_namespace[(iPS.getParameter<std::string>("outputListVariable")).c_str()].cast<pybind11::list>();
    put_ = produces<int>();

    usesResource("python");
  }

  void PythonTestProducer::produce(edm::Event& iEvent, edm::EventSetup const&) {
    edm::Handle<IntProduct> h;
    iEvent.getByToken(get_, h);
    {
      pybind11::gil_scoped_acquire acquire;
      outputList_.append(h->value);
    }
    iEvent.emplace(put_, h->value + value_);
  }
}  // namespace edmtest

//define this as a plug-in
DEFINE_FWK_MODULE(edmtest::PythonTestProducer);
