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
  class FillerBase {
  public:
    virtual ~FillerBase()  = default;
    virtual void fill() = 0;
    virtual void reset() = 0;
  };
  
  class TH1FFiller : public FillerBase {
  public:
    TH1FFiller(const edm::ParameterSet& iPSet,DQMStore& iStore,bool iSetLumiFlag):
    m_min(iPSet.getUntrackedParameter<double>("lowX")),
    m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
      std::string extension;
      if(iSetLumiFlag) {
        extension = "_lumi";
      }
      m_element = iStore.book1D(iPSet.getUntrackedParameter<std::string>("name")+extension,
                                              iPSet.getUntrackedParameter<std::string>("title")+extension,
                                              m_steps,
                                              m_min,
                                              iPSet.getUntrackedParameter<double>("highX"));
      if(iSetLumiFlag) {
        m_element->setLumiFlag();
      }
      m_hist = m_element->getTH1F();
      m_valueToFill=iPSet.getUntrackedParameter<double>("value");
    }
   
    virtual ~TH1FFiller() {};
 
    void reset() {
      m_element->Reset();
    }
    void fill() {
      m_hist->Fill(m_valueToFill);
    }
  private:
    TH1F* m_hist;
    double m_valueToFill;
    double m_min;
    unsigned int m_steps;
    MonitorElement* m_element;
  };
  
  class TH2FFiller : public FillerBase {
  public:
    TH2FFiller(const edm::ParameterSet& iPSet,DQMStore& iStore,bool iSetLumiFlag):
    m_min(iPSet.getUntrackedParameter<double>("lowX")),
    m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
      std::string extension;
      if(iSetLumiFlag) {
        extension = "_lumi";
      }
      m_element = iStore.book2D(iPSet.getUntrackedParameter<std::string>("name")+extension,
                                iPSet.getUntrackedParameter<std::string>("title")+extension,
                                m_steps,
                                m_min,
                                iPSet.getUntrackedParameter<double>("highX"),
                                iPSet.getUntrackedParameter<int>("nchY"),
                                iPSet.getUntrackedParameter<double>("lowY"),
                                iPSet.getUntrackedParameter<double>("highY"));
      if(iSetLumiFlag) {
        m_element->setLumiFlag();
      }
      m_hist = m_element->getTH2F();
      m_valueToFill=iPSet.getUntrackedParameter<double>("value");
    }
    
    virtual ~TH2FFiller() {};
 
    void reset() {
      m_element->Reset();
    }
    void fill() {
      m_hist->Fill(m_valueToFill, m_valueToFill);
    }
  private:
    TH2F* m_hist;
    double m_valueToFill;
    double m_min;
    unsigned int m_steps;
    MonitorElement* m_element;
  };
  
}

class DummyFillDQMStore :  public edm::EDAnalyzer {
   public:
      explicit DummyFillDQMStore(const edm::ParameterSet&);
      ~DummyFillDQMStore();

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
      std::vector<boost::shared_ptr<FillerBase> > m_runFillers;
      std::vector<boost::shared_ptr<FillerBase> > m_lumiFillers;
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
DummyFillDQMStore::DummyFillDQMStore(const edm::ParameterSet& iConfig):
m_fillRuns(iConfig.getUntrackedParameter<bool>("fillRuns")),
m_fillLumis(iConfig.getUntrackedParameter<bool>("fillLumis"))
{
  edm::Service<DQMStore> dstore;

  typedef std::vector<edm::ParameterSet> PSets;
  const PSets& elements = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("elements");
  if(m_fillRuns) {
    m_runFillers.reserve(elements.size());
    for( PSets::const_iterator it = elements.begin(), itEnd = elements.end(); it != itEnd; ++it){
      switch(it->getUntrackedParameter<unsigned int>("type",1)) {
        case 1:
        m_runFillers.push_back(boost::shared_ptr<FillerBase>(new TH1FFiller(*it,*dstore,false)));
        break;
        case 2:
        m_runFillers.push_back(boost::shared_ptr<FillerBase>(new TH2FFiller(*it,*dstore,false)));
        break;
      }
    }
  }

  if(m_fillLumis) {
    m_lumiFillers.reserve(elements.size());
    for( PSets::const_iterator it = elements.begin(), itEnd = elements.end(); it != itEnd; ++it){
      switch(it->getUntrackedParameter<unsigned int>("type",1)) {
        case 1:
        m_lumiFillers.push_back(boost::shared_ptr<FillerBase>(new TH1FFiller(*it,*dstore,true)));
        break;
        case 2:
        m_lumiFillers.push_back(boost::shared_ptr<FillerBase>(new TH2FFiller(*it,*dstore,true)));
        break;
      }
    }
  }


}


DummyFillDQMStore::~DummyFillDQMStore()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyFillDQMStore::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
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
DummyFillDQMStore::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DummyFillDQMStore::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
DummyFillDQMStore::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
DummyFillDQMStore::endRun(edm::Run const&, edm::EventSetup const&)
{
  for(std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_runFillers.begin(), itEnd = m_runFillers.end();
  it != itEnd;
  ++it) {
    (*it)->fill();
  }
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
DummyFillDQMStore::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  for(std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin(), itEnd = m_lumiFillers.end();
  it != itEnd;
  ++it) {
    (*it)->reset();
  }
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
DummyFillDQMStore::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  for(std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin(), itEnd = m_lumiFillers.end();
  it != itEnd;
  ++it) {
    (*it)->fill();
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyFillDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyFillDQMStore);
