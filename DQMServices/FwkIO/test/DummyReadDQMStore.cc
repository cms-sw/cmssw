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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
namespace {
  class ReaderBase {
  public:
    virtual ~ReaderBase()  = default;
    virtual void read() = 0;
  };
  
  class TH1FReader : public ReaderBase {
  public:
    TH1FReader(const edm::ParameterSet& iPSet,DQMStore& iStore,bool iSetLumiFlag):
    m_store(&iStore),
    m_element(0),
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
   
    virtual ~TH1FReader() {};
 
    void read() {
      if(0==m_element) {
        m_element = m_store->get(m_name);
        if(0==m_element) {
          throw cms::Exception("MissingElement")<<"The element: "<<m_name<<" was not found";
        }
      }
      TH1F* hist = m_element->getTH1F();
      assert(m_indexToCheck < m_means.size());
      
      if(hist->GetEntries() != m_entries[m_indexToCheck]) {
        throw cms::Exception("WrongEntries")<<"The element: "<<m_name<<" for index "<<m_indexToCheck
          <<" was expected to have "<<m_entries[m_indexToCheck]<<" entries but instead has "<<hist->GetEntries();
      }

      if(hist->GetMean() != m_means[m_indexToCheck]) {
        throw cms::Exception("WrongEntries")<<"The element: "<<m_name<<" for index "<<m_indexToCheck
          <<" was expected to have "<<m_means[m_indexToCheck]<<" mean but instead has "<<hist->GetMean();
      }
      
      ++m_indexToCheck;
    }
  private:
    std::string m_name;
    DQMStore* m_store;
    MonitorElement* m_element;
    std::vector<double> m_means;
    std::vector<double> m_entries;
    unsigned int m_indexToCheck;
  };
  
  class TH2FReader : public ReaderBase {
  public:
    TH2FReader(const edm::ParameterSet& iPSet,DQMStore& iStore,bool iSetLumiFlag):
    m_store(&iStore),
    m_element(0),
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
   
    virtual ~TH2FReader() {};
 
    void read() {
      if(0==m_element) {
        m_element = m_store->get(m_name);
        if(0==m_element) {
          throw cms::Exception("MissingElement")<<"The element: "<<m_name<<" was not found";
        }
      }
      TH2F* hist = m_element->getTH2F();
      assert(m_indexToCheck < m_means.size());
      
      if(hist->GetEntries() != m_entries[m_indexToCheck]) {
        throw cms::Exception("WrongEntries")<<"The element: "<<m_name<<" for index "<<m_indexToCheck
          <<" was expected to have "<<m_entries[m_indexToCheck]<<" entries but instead has "<<hist->GetEntries();
      }

      if(hist->GetMean() != m_means[m_indexToCheck]) {
        throw cms::Exception("WrongMeans")<<"The element: "<<m_name<<" for index "<<m_indexToCheck
          <<" was expected to have "<<m_means[m_indexToCheck]<<" mean but instead has "<<hist->GetMean();
      }
      
      ++m_indexToCheck;
    }
  private:
    std::string m_name;
    DQMStore* m_store;
    MonitorElement* m_element;
    std::vector<double> m_means;
    std::vector<double> m_entries;
    unsigned int m_indexToCheck;
  };
  
}

class DummyReadDQMStore :  public edm::EDAnalyzer {
   public:
      explicit DummyReadDQMStore(const edm::ParameterSet&);
      ~DummyReadDQMStore();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
      std::vector<boost::shared_ptr<ReaderBase> > m_runReaders;
      std::vector<boost::shared_ptr<ReaderBase> > m_lumiReaders;
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
DummyReadDQMStore::DummyReadDQMStore(const edm::ParameterSet& iConfig)
{
  edm::Service<DQMStore> dstore;

  typedef std::vector<edm::ParameterSet> PSets;
  const PSets& runElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("runElements");
  m_runReaders.reserve(runElements.size());
  for( PSets::const_iterator it = runElements.begin(), itEnd = runElements.end(); it != itEnd; ++it){
    switch(it->getUntrackedParameter<unsigned int>("type",1)) {
      case 1:
      m_runReaders.push_back(boost::shared_ptr<ReaderBase>(new TH1FReader(*it,*dstore,false)));
      break;
      case 2:
      m_runReaders.push_back(boost::shared_ptr<ReaderBase>(new TH2FReader(*it,*dstore,false)));
      break;
    }
  }

  const PSets& lumiElements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("lumiElements");
  m_lumiReaders.reserve(lumiElements.size());
  for( PSets::const_iterator it = lumiElements.begin(), itEnd = lumiElements.end(); it != itEnd; ++it){
    switch(it->getUntrackedParameter<unsigned int>("type",1)) {
      case 1:
      m_lumiReaders.push_back(boost::shared_ptr<ReaderBase>(new TH1FReader(*it,*dstore,true)));
      break;
      case 2:
      m_lumiReaders.push_back(boost::shared_ptr<ReaderBase>(new TH2FReader(*it,*dstore,true)));
      break;
    }
  }


}


DummyReadDQMStore::~DummyReadDQMStore()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyReadDQMStore::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{
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
void 
DummyReadDQMStore::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DummyReadDQMStore::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
DummyReadDQMStore::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
DummyReadDQMStore::endRun(edm::Run const&, edm::EventSetup const&)
{
  for(std::vector<boost::shared_ptr<ReaderBase> >::iterator it = m_runReaders.begin(), itEnd = m_runReaders.end();
  it != itEnd;
  ++it) {
    (*it)->read();
  }
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
DummyReadDQMStore::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
DummyReadDQMStore::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  for(std::vector<boost::shared_ptr<ReaderBase> >::iterator it = m_lumiReaders.begin(), itEnd = m_lumiReaders.end();
  it != itEnd;
  ++it) {
    (*it)->read();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyReadDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyReadDQMStore);
