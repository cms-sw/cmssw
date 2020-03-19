#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include <boost/python.hpp>
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
    boost::python::list outputList_;
  };

  PythonTestProducer::PythonTestProducer(edm::ParameterSet const& iPS)
      : get_(consumes<IntProduct>(iPS.getParameter<edm::InputTag>("source"))) {
    using namespace boost::python;
    object main_module{
        boost::python::handle<>(boost::python::borrowed(PyImport_AddModule(const_cast<char*>("__main__"))))};
    auto main_namespace = main_module.attr("__dict__");

    //NOTE attempts to hold the object directly and read it in `produce` lead to segmentation faults
    value_ = extract<int>(main_namespace[iPS.getParameter<std::string>("inputVariable")]);

    outputList_ = extract<list>(main_namespace[iPS.getParameter<std::string>("outputListVariable")]);

    put_ = produces<int>();

    usesResource("python");
  }

  void PythonTestProducer::produce(edm::Event& iEvent, edm::EventSetup const&) {
    using namespace boost::python;

    edm::Handle<IntProduct> h;
    iEvent.getByToken(get_, h);

    outputList_.append(h->value);

    iEvent.emplace(put_, h->value + value_);
  }
}  // namespace edmtest

//define this as a plug-in
DEFINE_FWK_MODULE(edmtest::PythonTestProducer);
