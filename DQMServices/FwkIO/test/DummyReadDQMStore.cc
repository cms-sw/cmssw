// -*- C++ -*-
//
// Package:    DummyReadDQMStore
// Class:      DummyReadDQMStore
//
/**\class DummyReadDQMStore DummyReadDQMStore.cc DQMServices/DummyReadDQMStore/src/DummyReadDQMStore.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Fri Apr 29 18:05:50 CDT 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
namespace {
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  class ReaderBase {
  public:
    virtual ~ReaderBase() = default;
    virtual void read(int run, int lumi) = 0;
  };

  class TH1Reader : public ReaderBase {
  public:
    TH1Reader(const edm::ParameterSet& iPSet, DQMStore& iStore, bool iSetLumiFlag)
        : m_store(&iStore),
          m_element(nullptr),
          m_runs(iPSet.getUntrackedParameter<std::vector<int> >("runs")),
          m_lumis(iPSet.getUntrackedParameter<std::vector<int> >("lumis")),
          m_means(iPSet.getUntrackedParameter<std::vector<double> >("means")),
          m_entries(iPSet.getUntrackedParameter<std::vector<double> >("entries")) {
      assert(m_means.size() == m_entries.size());
      std::string extension;
      if (iSetLumiFlag) {
        extension = "_lumi";
      }
      m_name = iPSet.getUntrackedParameter<std::string>("name") + extension;
    }

    ~TH1Reader() override{};

    void read(int run, int lumi) override {
      double expected_mean = -1, expected_entries = -1;
      for (unsigned int i = 0; i < m_runs.size(); i++) {
        if (m_runs[i] == run && m_lumis[i] == lumi) {
          expected_mean = m_means[i];
          expected_entries = m_entries[i];
        }
      }
      assert(expected_entries != -1 || !"Unexpected run/lumi!");

      m_element = m_store->get(m_name);
      if (nullptr == m_element) {
        throw cms::Exception("MissingElement") << "The element: " << m_name << " was not found";
      }
      TH1* hist = m_element->getTH1();

      if (hist->GetEntries() != expected_entries) {
        throw cms::Exception("WrongEntries")
            << "The element: " << m_name << " for run " << run << " lumi " << lumi << " was expected  to have "
            << expected_entries << " entries but instead has " << hist->GetEntries();
      }

      if (hist->GetMean() != expected_mean) {
        throw cms::Exception("WrongEntries")
            << "The element: " << m_name << " for run " << run << " lumi " << lumi << " was expected  to have "
            << expected_mean << " mean but instead has " << hist->GetMean();
      }
    }

  private:
    std::string m_name;
    DQMStore* m_store;
    MonitorElement* m_element;
    std::vector<int> m_runs;
    std::vector<int> m_lumis;
    std::vector<double> m_means;
    std::vector<double> m_entries;
  };

}  // namespace

class DummyReadDQMStore : public edm::EDAnalyzer {
public:
  explicit DummyReadDQMStore(const edm::ParameterSet&);
  ~DummyReadDQMStore() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  std::vector<std::shared_ptr<ReaderBase> > m_runReaders;
  std::vector<std::shared_ptr<ReaderBase> > m_lumiReaders;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DummyReadDQMStore::DummyReadDQMStore(const edm::ParameterSet& iConfig) {
  edm::Service<DQMStore> dstore;

  typedef std::vector<edm::ParameterSet> PSets;
  const PSets& runElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("runElements");
  m_runReaders.reserve(runElements.size());
  for (PSets::const_iterator it = runElements.begin(), itEnd = runElements.end(); it != itEnd; ++it) {
    switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
      case 1:
        m_runReaders.push_back(std::shared_ptr<ReaderBase>(new TH1Reader(*it, *dstore, false)));
        break;
      case 2:
        m_runReaders.push_back(std::shared_ptr<ReaderBase>(new TH1Reader(*it, *dstore, false)));
        break;
    }
  }

  const PSets& lumiElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("lumiElements");
  m_lumiReaders.reserve(lumiElements.size());
  for (PSets::const_iterator it = lumiElements.begin(), itEnd = lumiElements.end(); it != itEnd; ++it) {
    switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
      case 1:
        m_lumiReaders.push_back(std::shared_ptr<ReaderBase>(new TH1Reader(*it, *dstore, true)));
        break;
      case 2:
        m_lumiReaders.push_back(std::shared_ptr<ReaderBase>(new TH1Reader(*it, *dstore, true)));
        break;
    }
  }
}

DummyReadDQMStore::~DummyReadDQMStore() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DummyReadDQMStore::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  using namespace edm;
  /* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::unique_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

  /* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
}

// ------------ method called once each job just before starting event loop  ------------
void DummyReadDQMStore::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DummyReadDQMStore::endJob() {}

// ------------ method called when starting to processes a run  ------------
void DummyReadDQMStore::beginRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a run  ------------
void DummyReadDQMStore::endRun(edm::Run const& run, edm::EventSetup const&) {
  for (std::vector<std::shared_ptr<ReaderBase> >::iterator it = m_runReaders.begin(), itEnd = m_runReaders.end();
       it != itEnd;
       ++it) {
    (*it)->read(run.run(), 0);
  }
}

// ------------ method called when starting to processes a luminosity block  ------------
void DummyReadDQMStore::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void DummyReadDQMStore::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
  for (std::vector<std::shared_ptr<ReaderBase> >::iterator it = m_lumiReaders.begin(), itEnd = m_lumiReaders.end();
       it != itEnd;
       ++it) {
    (*it)->read(lumi.run(), lumi.luminosityBlock());
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DummyReadDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyReadDQMStore);
