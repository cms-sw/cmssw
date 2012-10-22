// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      SiStripQualityHistory
// 
/**\class SiStripQualityHistory SiStripQualityHistory.cc DPGAnalysis/SiStripTools/plugins/SiStripQualityHistory.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Sep 18 17:52:00 CEST 2009
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>

//#include "TGraph.h"
#include "TH1F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

//
// class decleration
//

class SiStripQualityHistory : public edm::EDAnalyzer {
 public:
    explicit SiStripQualityHistory(const edm::ParameterSet&);
    ~SiStripQualityHistory();

  enum {Module,Fiber,APV};

   private:
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  const std::vector<edm::ParameterSet> _monitoredssq;
  const unsigned int _mode;
  //  std::map<std::string,TGraph*> _history;
  std::map<std::string,TH1F*> _history;


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
SiStripQualityHistory::SiStripQualityHistory(const edm::ParameterSet& iConfig):
  _monitoredssq(iConfig.getParameter<std::vector<edm::ParameterSet> >("monitoredSiStripQuality")),
  _mode(iConfig.getUntrackedParameter<unsigned int>("granularityMode",Module)),
  _history()
{
   //now do what ever initialization is needed

  edm::Service<TFileService> tfserv;

  for(std::vector<edm::ParameterSet>::const_iterator ps=_monitoredssq.begin();ps!=_monitoredssq.end();++ps) {
    
    std::string name = ps->getParameter<std::string>("name");
    //    _history[name] = tfserv->make<TGraph>();
    //    _history[name]->SetName(name.c_str());     _history[name]->SetTitle(name.c_str()); 

    _history[name] = tfserv->make<TH1F>(name.c_str(),name.c_str(),10,0,10);

  }

}


SiStripQualityHistory::~SiStripQualityHistory()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiStripQualityHistory::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("EventProcessing") << "event being processed";
}

void 
SiStripQualityHistory::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  // loop on all the SiStripQuality objects to be monitored

  for(std::vector<edm::ParameterSet>::const_iterator ps=_monitoredssq.begin();ps!=_monitoredssq.end();++ps) {
    
    std::string name = ps->getParameter<std::string>("name");
    std::string label = ps->getParameter<std::string>("ssqLabel");

    
    edm::ESHandle<SiStripQuality> ssq;
    iSetup.get<SiStripQualityRcd>().get(label,ssq);

    std::vector<SiStripQuality::BadComponent> bads = ssq->getBadComponentList();

    edm::LogInfo("Debug") << bads.size() << " bad components found";

    int nbad=0;

    for(std::vector<SiStripQuality::BadComponent>::const_iterator bc=bads.begin();bc!=bads.end();++bc) {

      if(_mode==Module) {
       if(bc->BadModule) ++nbad;
      }
      else if(_mode == Fiber) {
      for(int fiber=1;fiber<5;fiber*=2) {
	if((bc->BadFibers & fiber)>0) ++nbad;
      }
    }
    else if(_mode ==APV) {
      for(int apv=1;apv<33;apv*=2) {
	if((bc->BadApvs & apv)>0) ++nbad;
      }
    }
  }


    //    _history[name]->SetPoint(_history[name]->GetN(),iRun.run(),nbad);
    char runname[100];
    sprintf(runname,"%d",iRun.run());
    edm::LogInfo("Debug") << name << " " << runname << " " << nbad;
    _history[name]->Fill(runname,nbad);

  }

  

}

void 
SiStripQualityHistory::endRun(const edm::Run& iRun, const edm::EventSetup&)
{}


// ------------ method called once each job just before starting event loop  ------------
void 
SiStripQualityHistory::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripQualityHistory::endJob() 
{
  /*
  for(std::vector<edm::ParameterSet>::const_iterator ps=_monitoredssq.begin();ps!=_monitoredssq.end();++ps) {
    
    std::string name = ps->getParameter<std::string>("name");
    _history[name]->Write();

  }
  */
}


//define this as a plug-in
DEFINE_FWK_MODULE(SiStripQualityHistory);
