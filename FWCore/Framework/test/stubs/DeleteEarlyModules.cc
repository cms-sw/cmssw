// -*- C++ -*-
//
// Package:     test
// Class  :     DeleteEarlyModules
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue Feb  7 15:36:37 CST 2012
//

// system include files
#include <vector>
#include <memory>

// user include files
#include "DataFormats/TestObjects/interface/DeleteEarly.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

namespace edmtest {
  class DeleteEarlyProducer : public edm::EDProducer {
  public:
    DeleteEarlyProducer(edm::ParameterSet const& pset) { produces<DeleteEarly>(); }

    virtual void beginJob() {
      // Needed because DeleteEarly objects may be allocated and deleted in initialization
      edmtest::DeleteEarly::resetDeleteCount();
    }

    virtual void produce(edm::Event& e, edm::EventSetup const&) { e.put(std::make_unique<DeleteEarly>()); }
  };

  class DeleteEarlyReader : public edm::global::EDAnalyzer<> {
  public:
    DeleteEarlyReader(edm::ParameterSet const& pset)
        : getToken_(consumes<DeleteEarly>(pset.getUntrackedParameter<edm::InputTag>("tag"))) {}

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const&) const override { e.get(getToken_); }

  private:
    edm::EDGetTokenT<DeleteEarly> getToken_;
  };

  class DeleteEarlyConsumer : public edm::global::EDAnalyzer<> {
  public:
    DeleteEarlyConsumer(edm::ParameterSet const& pset) {
      consumes<DeleteEarly>(pset.getUntrackedParameter<edm::InputTag>("tag"));
    }

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {}

  private:
  };

  class DeleteEarlyCheckDeleteAnalyzer : public edm::EDAnalyzer {
  public:
    DeleteEarlyCheckDeleteAnalyzer(edm::ParameterSet const& pset)
        : m_expectedValues(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedValues")), m_index(0) {}

    virtual void analyze(edm::Event const&, edm::EventSetup const&) {
      if (DeleteEarly::nDeletes() != m_expectedValues.at(m_index)) {
        throw cms::Exception("DeleteEarlyError")
            << "On index " << m_index << " we expected " << m_expectedValues[m_index] << " deletes but we see "
            << DeleteEarly::nDeletes();
      }
      ++m_index;
    }

  private:
    std::vector<unsigned int> m_expectedValues;
    unsigned int m_index;
  };
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(DeleteEarlyProducer);
DEFINE_FWK_MODULE(DeleteEarlyReader);
DEFINE_FWK_MODULE(DeleteEarlyConsumer);
DEFINE_FWK_MODULE(DeleteEarlyCheckDeleteAnalyzer);
