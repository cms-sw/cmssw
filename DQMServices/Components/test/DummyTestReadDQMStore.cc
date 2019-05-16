// system include files
#include <memory>
#include <utility>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
namespace {
class ReaderBase {
 public:
  virtual ~ReaderBase()  = default;
  virtual void read() = 0;
  virtual void reset() = 0;
};
  
class TH1FReader : public ReaderBase {
 public:
  TH1FReader(const edm::ParameterSet& iPSet,
             DQMStore& iStore,
             std::string folder,
             bool iSetLumiFlag)
      : folder_(std::move(std::move(folder))),
        m_store(&iStore),
        m_element(nullptr),
        m_means(iPSet.getUntrackedParameter<std::vector<double> >("means")),
        m_entries(iPSet.getUntrackedParameter<std::vector<double> >("entries")),
        m_indexToCheck(0) {
    assert(m_means.size() == m_entries.size());
    std::string extension;
    if(iSetLumiFlag) {
      extension = "_lumi";
    }
    m_name = iPSet.getUntrackedParameter<std::string>("name")+extension;
  }
   
  ~TH1FReader() override = default;;
 
  void reset() override {
    if (nullptr == m_element) {
      m_element = m_store->get(folder_ + m_name);
      if (nullptr != m_element)
        m_element->Reset();
    }
  }
    
  void read() override {
    if (nullptr == m_element) {
      m_element = m_store->get(folder_ + m_name);
      if (nullptr == m_element) {
        throw cms::Exception("MissingElement") << "The element: "
                                               << m_name << " was not found";
      }
    }
    TH1F* hist = m_element->getTH1F();
    assert(m_indexToCheck < m_means.size());
      
    if(hist->GetEntries() != m_entries[m_indexToCheck]) {
      throw cms::Exception("WrongEntries")<< "The element: " << m_name
                                          << " for index " << m_indexToCheck
                                          << " was expected to have "
                                          << m_entries[m_indexToCheck]
                                          << " entries but instead has "
                                          << hist->GetEntries();
    }

    if(hist->GetMean() != m_means[m_indexToCheck]) {
      throw cms::Exception("WrongEntries") << "The element: " << m_name
                                           << " for index " << m_indexToCheck
                                           << " was expected to have "
                                           << m_means[m_indexToCheck]
                                           << " mean but instead has "
                                           << hist->GetMean();
    }
    ++m_indexToCheck;
  }
 private:
  std::string folder_;
  std::string m_name;
  DQMStore* m_store;
  MonitorElement* m_element;
  std::vector<double> m_means;
  std::vector<double> m_entries;
  unsigned int m_indexToCheck;
};
  
class TH2FReader : public ReaderBase {
 public:
  TH2FReader(const edm::ParameterSet& iPSet,DQMStore& iStore, std::string folder, bool iSetLumiFlag):
      folder_(std::move(std::move(folder))),
      m_store(&iStore),
      m_element(nullptr),
      m_means(iPSet.getUntrackedParameter<std::vector<double> >("means")),
      m_entries(iPSet.getUntrackedParameter<std::vector<double> >("entries")),
      m_indexToCheck(0)
  {
    assert(m_means.size() == m_entries.size());
    std::string extension;
    if(iSetLumiFlag) {
      extension = "_lumi";
    }
    m_name = iPSet.getUntrackedParameter<std::string>("name")+extension;
  }
   
  ~TH2FReader() override = default;;
 
  void reset() override {
    if (nullptr == m_element) {
      m_element = m_store->get(folder_ + m_name);
      if (nullptr != m_element)
        m_element->Reset();
    }
  }

  void read() override {
    if (nullptr == m_element) {
      m_element = m_store->get(folder_ + m_name);
      if (nullptr == m_element) {
        throw cms::Exception("MissingElement") << "The element: " << m_name
                                               << " was not found";
      }
    }
    TH2F* hist = m_element->getTH2F();
    assert(m_indexToCheck < m_means.size());
      
    if (hist->GetEntries() != m_entries[m_indexToCheck]) {
      throw cms::Exception("WrongEntries") << "The element: " << m_name
                                           << " for index " << m_indexToCheck
                                           << " was expected to have "
                                           << m_entries[m_indexToCheck]
                                           << " entries but instead has "
                                           << hist->GetEntries();
    }

    if(hist->GetMean() != m_means[m_indexToCheck]) {
      throw cms::Exception("WrongMeans") << "The element: " << m_name
                                         << " for index " << m_indexToCheck
                                         << " was expected to have "
                                         << m_means[m_indexToCheck]
                                         << " mean but instead has "
                                         << hist->GetMean();
    }
    ++m_indexToCheck;
  }
 private:
  std::string folder_;
  std::string m_name;
  DQMStore* m_store;
  MonitorElement* m_element;
  std::vector<double> m_means;
  std::vector<double> m_entries;
  unsigned int m_indexToCheck;
};
  
}

class DummyTestReadDQMStore :  public edm::EDAnalyzer {
 public:
  explicit DummyTestReadDQMStore(const edm::ParameterSet&);
  ~DummyTestReadDQMStore() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
      
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  std::vector<boost::shared_ptr<ReaderBase> > m_runReaders;
  std::vector<boost::shared_ptr<ReaderBase> > m_lumiReaders;
  std::string folder_;
  unsigned int runToCheck_;
  using PSets = std::vector<edm::ParameterSet>;
  PSets runElements;
  PSets lumiElements;
  edm::Service<DQMStore> dstore;
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
DummyTestReadDQMStore::DummyTestReadDQMStore(const edm::ParameterSet& iConfig)
{
  folder_ = iConfig.getUntrackedParameter<std::string >("folder", "TestFolder/");
  runToCheck_ = iConfig.getUntrackedParameter<int>("runToCheck", 1);
  runElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("runElements");
  lumiElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("lumiElements");
}


DummyTestReadDQMStore::~DummyTestReadDQMStore()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyTestReadDQMStore::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{
  using namespace edm;
  /* This is an event example
  //Read 'ExampleData' from the Event
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);

  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  std::unique_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
  iEvent.put(std::move(pOut));
  */

  /* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
  */
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
DummyTestReadDQMStore::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DummyTestReadDQMStore::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
DummyTestReadDQMStore::beginRun(edm::Run const& iRun, edm::EventSetup const&)
{
  if(iRun.run() != runToCheck_) return;

  m_runReaders.reserve(runElements.size());
  for( auto it = runElements.begin(), itEnd = runElements.end(); it != itEnd; ++it) {
    switch(it->getUntrackedParameter<unsigned int>("type",1)) {
    case 1:
      m_runReaders.push_back(boost::shared_ptr<ReaderBase>(new TH1FReader(*it,*dstore,folder_,false)));
      break;
    case 2:
      m_runReaders.push_back(boost::shared_ptr<ReaderBase>(new TH2FReader(*it,*dstore,folder_,false)));
      break;
    }
  }
  m_lumiReaders.reserve(lumiElements.size());
  for( auto it = lumiElements.begin(), itEnd = lumiElements.end(); it != itEnd; ++it){
    switch(it->getUntrackedParameter<unsigned int>("type",1)) {
      case 1:
        m_lumiReaders.push_back(boost::shared_ptr<ReaderBase>(new TH1FReader(*it,*dstore,folder_,true)));
        break;
      case 2:
        m_lumiReaders.push_back(boost::shared_ptr<ReaderBase>(new TH2FReader(*it,*dstore,folder_,true)));
        break;
    }
  }
}

// ------------ method called when ending the processing of a run  ------------
void 
DummyTestReadDQMStore::endRun(edm::Run const& iRun, edm::EventSetup const&)
{
  if(iRun.run() != runToCheck_) return;

  for(auto & m_runReader : m_runReaders) 
    {
      m_runReader->read();
    }
  m_runReaders.erase(m_runReaders.begin(), m_runReaders.end());
  m_lumiReaders.erase(m_lumiReaders.begin(), m_lumiReaders.end());

  std::cout << "checked Run: " << iRun.run() << std::endl;

}

// ------------ method called when starting to processes a luminosity block  ------------
void 
DummyTestReadDQMStore::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  //   for(std::vector<boost::shared_ptr<ReaderBase> >::iterator it = m_lumiReaders.begin(), itEnd = m_lumiReaders.end();
  //   it != itEnd;
  //   ++it) {
  //     (*it)->reset();
  //   }
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
DummyTestReadDQMStore::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  for(auto & m_lumiReader : m_lumiReaders) {
    m_lumiReader->read();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyTestReadDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyTestReadDQMStore);
