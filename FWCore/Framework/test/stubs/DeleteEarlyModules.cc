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
// $Id: DeleteEarlyModules.cc,v 1.1 2012/02/09 22:12:57 chrjones Exp $
//

// system include files
#include <vector>
#include <memory>

// user include files
#include "DataFormats/TestObjects/interface/DeleteEarly.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"


namespace edmtest {
  class DeleteEarlyProducer: public edm::EDProducer {
  public:
    DeleteEarlyProducer(edm::ParameterSet const& pset) {
      produces<DeleteEarly>();
    }
    
    virtual void produce(edm::Event& e, edm::EventSetup const& ){
      std::auto_ptr<DeleteEarly> p(new DeleteEarly);
      e.put(p);
    }

  };
  
  class DeleteEarlyReader: public edm::EDAnalyzer {
  public:
    DeleteEarlyReader(edm::ParameterSet const& pset):
    m_tag(pset.getUntrackedParameter<edm::InputTag>("tag"))
    {}
    
    virtual void analyze(edm::Event const& e, edm::EventSetup const& ) {
      edm::Handle<DeleteEarly> h;
      e.getByLabel(m_tag,h);
    }
  private:
    edm::InputTag m_tag;
  };
  
  class DeleteEarlyCheckDeleteAnalyzer : public edm::EDAnalyzer {
  public:
    DeleteEarlyCheckDeleteAnalyzer(edm::ParameterSet const& pset):
    m_expectedValues(pset.getUntrackedParameter<std::vector<unsigned int>>("expectedValues")),
    m_index(0)
    {}
    
    virtual void analyze(edm::Event const&, edm::EventSetup const&) {
      if (DeleteEarly::nDeletes() != m_expectedValues.at(m_index)) {
        throw cms::Exception("DeleteEarlyError")<<"On index "<<m_index<<" we expected "<<m_expectedValues[m_index]<<" deletes but we see "<<DeleteEarly::nDeletes();
      }
      ++m_index;
    }

  private: 
    std::vector<unsigned int> m_expectedValues;
    unsigned int m_index;
  };
}
using namespace edmtest;
DEFINE_FWK_MODULE(DeleteEarlyProducer);
DEFINE_FWK_MODULE(DeleteEarlyReader);
DEFINE_FWK_MODULE(DeleteEarlyCheckDeleteAnalyzer);

