/* This was written to benchmark some changes to
the getByLabel function and supporting code. It makes
a lot of getByLabel calls although it is not particularly
realistic ... */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>

namespace edmtest {

  class ManyProductProducer : public edm::EDProducer {
  public:
    explicit ManyProductProducer(edm::ParameterSet const& iConfig);

    ~ManyProductProducer() override;

    void produce(edm::Event& e, edm::EventSetup const& c) override;

  private:
    unsigned int nProducts_;
    std::vector<std::string> instanceNames_;
  };

  ManyProductProducer::ManyProductProducer(edm::ParameterSet const& iConfig)
      : nProducts_(iConfig.getUntrackedParameter<unsigned int>("nProducts", 1)) {
    for (unsigned int i = 0; i < nProducts_; ++i) {
      std::stringstream instanceName;
      instanceName << "i" << i;
      instanceNames_.push_back(instanceName.str());
      produces<IntProduct>(instanceName.str());
    }
  }

  ManyProductProducer::~ManyProductProducer() {}

  // Functions that gets called by framework every event
  void ManyProductProducer::produce(edm::Event& e, edm::EventSetup const&) {
    for (unsigned int i = 0; i < nProducts_; ++i) {
      e.put(std::make_unique<IntProduct>(1), instanceNames_[i]);
    }
  }

  class ManyProductAnalyzer : public edm::EDAnalyzer {
  public:
    explicit ManyProductAnalyzer(edm::ParameterSet const& iConfig);

    ~ManyProductAnalyzer() override;

    void analyze(edm::Event const&, edm::EventSetup const&) override;

  private:
    unsigned int nProducts_;
    std::vector<edm::InputTag> tags_;
  };

  ManyProductAnalyzer::ManyProductAnalyzer(edm::ParameterSet const& iConfig)
      : nProducts_(iConfig.getUntrackedParameter<unsigned int>("nProducts", 1)) {
    for (unsigned int i = 0; i < nProducts_; ++i) {
      std::stringstream instanceName;
      instanceName << "i" << i;
      edm::InputTag tag("produceInts", instanceName.str());
      tags_.push_back(tag);
    }
  }

  ManyProductAnalyzer::~ManyProductAnalyzer() {}

  void ManyProductAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    edm::Handle<IntProduct> h;
    for (auto const& tag : tags_) {
      e.getByLabel(tag, h);
      if (!h.isValid()) {
        abort();
      }
    }
  }
}  // namespace edmtest

using edmtest::ManyProductProducer;
DEFINE_FWK_MODULE(ManyProductProducer);

using edmtest::ManyProductAnalyzer;
DEFINE_FWK_MODULE(ManyProductAnalyzer);
