// -*- C++ -*-
//
// Package:    DummyFillDQMStore
// Class:      DummyFillDQMStore
//
/**\class DummyFillDQMStore DummyFillDQMStore.cc DQMServices/DummyFillDQMStore/src/DummyFillDQMStore.cc

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
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
namespace {
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  class FillerBase {
  public:
    virtual ~FillerBase() = default;
    virtual void fill() = 0;
    virtual void reset() = 0;
  };

  class TH1FFiller : public FillerBase {
  public:
    TH1FFiller(const edm::ParameterSet& iPSet, DQMStore& iStore, bool iSetLumiFlag)
        : m_min(iPSet.getUntrackedParameter<double>("lowX")), m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
      std::string extension;
      if (iSetLumiFlag) {
        extension = "_lumi";
      }
      m_element = iStore.book1D(iPSet.getUntrackedParameter<std::string>("name") + extension,
                                iPSet.getUntrackedParameter<std::string>("title") + extension,
                                m_steps,
                                m_min,
                                iPSet.getUntrackedParameter<double>("highX"));
      m_valueToFill = iPSet.getUntrackedParameter<double>("value");
    }

    virtual ~TH1FFiller(){};

    void reset() { m_element->Reset(); }
    void fill() { m_element->Fill(m_valueToFill); }

  private:
    double m_valueToFill;
    double m_min;
    unsigned int m_steps;
    MonitorElement* m_element;
  };

  class TH2FFiller : public FillerBase {
  public:
    TH2FFiller(const edm::ParameterSet& iPSet, DQMStore& iStore, bool iSetLumiFlag)
        : m_min(iPSet.getUntrackedParameter<double>("lowX")), m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
      std::string extension;
      if (iSetLumiFlag) {
        extension = "_lumi";
      }
      m_element = iStore.book2D(iPSet.getUntrackedParameter<std::string>("name") + extension,
                                iPSet.getUntrackedParameter<std::string>("title") + extension,
                                m_steps,
                                m_min,
                                iPSet.getUntrackedParameter<double>("highX"),
                                iPSet.getUntrackedParameter<int>("nchY"),
                                iPSet.getUntrackedParameter<double>("lowY"),
                                iPSet.getUntrackedParameter<double>("highY"));
      m_valueToFill = iPSet.getUntrackedParameter<double>("value");
    }

    virtual ~TH2FFiller(){};

    void reset() { m_element->Reset(); }
    void fill() { m_element->Fill(m_valueToFill, m_valueToFill); }

  private:
    double m_valueToFill;
    double m_min;
    unsigned int m_steps;
    MonitorElement* m_element;
  };

}  // namespace

class DummyFillDQMStore : public DQMOneLumiEDAnalyzer<> {
public:
  explicit DummyFillDQMStore(const edm::ParameterSet&);
  ~DummyFillDQMStore();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void dqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------
  edm::ParameterSet iConfig;
  std::vector<std::shared_ptr<FillerBase> > m_runFillers;
  std::vector<std::shared_ptr<FillerBase> > m_lumiFillers;
  bool m_fillRuns;
  bool m_fillLumis;
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
DummyFillDQMStore::DummyFillDQMStore(const edm::ParameterSet& iConfig)
    : m_fillRuns(iConfig.getUntrackedParameter<bool>("fillRuns")),
      m_fillLumis(iConfig.getUntrackedParameter<bool>("fillLumis")) {
  this->iConfig = iConfig;
}

void DummyFillDQMStore::bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) {
  // this is fine as long as we don't do concurrent booking.
  edm::Service<DQMStore> dstore;

  typedef std::vector<edm::ParameterSet> PSets;
  const PSets& elements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("elements");
  if (m_fillRuns) {
    dstore->setScope(MonitorElementData::Scope::RUN);
    m_runFillers.clear();
    m_runFillers.reserve(elements.size());
    for (PSets::const_iterator it = elements.begin(), itEnd = elements.end(); it != itEnd; ++it) {
      switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
        case 1:
          m_runFillers.push_back(std::shared_ptr<FillerBase>(new TH1FFiller(*it, *dstore, false)));
          break;
        case 2:
          m_runFillers.push_back(std::shared_ptr<FillerBase>(new TH2FFiller(*it, *dstore, false)));
          break;
      }
    }
  }

  if (m_fillLumis) {
    dstore->setScope(MonitorElementData::Scope::LUMI);
    m_lumiFillers.clear();
    m_lumiFillers.reserve(elements.size());
    for (PSets::const_iterator it = elements.begin(), itEnd = elements.end(); it != itEnd; ++it) {
      switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
        case 1:
          m_lumiFillers.push_back(std::shared_ptr<FillerBase>(new TH1FFiller(*it, *dstore, true)));
          break;
        case 2:
          m_lumiFillers.push_back(std::shared_ptr<FillerBase>(new TH2FFiller(*it, *dstore, true)));
          break;
      }
    }
  }
}

DummyFillDQMStore::~DummyFillDQMStore() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void DummyFillDQMStore::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
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

// ------------ method called when ending the processing of a run  ------------
void DummyFillDQMStore::dqmEndRun(edm::Run const&, edm::EventSetup const&) {
  for (std::vector<std::shared_ptr<FillerBase> >::iterator it = m_runFillers.begin(), itEnd = m_runFillers.end();
       it != itEnd;
       ++it) {
    (*it)->fill();
  }
}

// ------------ method called when starting to processes a luminosity block  ------------
void DummyFillDQMStore::dqmBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  for (std::vector<std::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin(), itEnd = m_lumiFillers.end();
       it != itEnd;
       ++it) {
    (*it)->reset();
  }
}

// ------------ method called when ending the processing of a luminosity block  ------------
void DummyFillDQMStore::dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  for (std::vector<std::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin(), itEnd = m_lumiFillers.end();
       it != itEnd;
       ++it) {
    (*it)->fill();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DummyFillDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyFillDQMStore);
