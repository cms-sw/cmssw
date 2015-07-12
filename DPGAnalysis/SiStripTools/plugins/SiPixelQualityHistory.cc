// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      SiPixelQualityHistory
//
/**\class SiPixelQualityHistory SiPixelQualityHistory.cc DPGAnalysis/SiStripTools/plugins/SiPixelQualityHistory.cc

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
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
//
// class decleration
//

class SiPixelQualityHistory : public edm::EDAnalyzer {
 public:
    explicit SiPixelQualityHistory(const edm::ParameterSet&);
    ~SiPixelQualityHistory();

  enum {Summary,Module,ROC};

   private:
      virtual void beginJob() override ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  RunHistogramManager m_rhm;
  const std::vector<edm::ParameterSet> m_monitoredspq;
  const unsigned int m_mode;
  const bool m_run;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  //  std::map<std::string,TGraph*> m_history;
  std::map<std::string,TH1F*> m_history;
  std::map<std::string,TProfile**> m_badmodrun;

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
SiPixelQualityHistory::SiPixelQualityHistory(const edm::ParameterSet& iConfig):
  m_rhm(consumesCollector()),
  m_monitoredspq(iConfig.getParameter<std::vector<edm::ParameterSet> >("monitoredSiPixelQuality")),
  m_mode(iConfig.getUntrackedParameter<unsigned int>("granularityMode",Module)),
  m_run(iConfig.getParameter<bool>("runProcess")),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",100)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",4)),
  m_history(),m_badmodrun()
{
   //now do what ever initialization is needed

  edm::Service<TFileService> tfserv;

  for(std::vector<edm::ParameterSet>::const_iterator ps=m_monitoredspq.begin();ps!=m_monitoredspq.end();++ps) {

    std::string name = ps->getParameter<std::string>("name");

    if(m_run) m_history[name] = tfserv->make<TH1F>(name.c_str(),name.c_str(),10,0,10);

    char hrunname[400];
    sprintf(hrunname,"badmodrun_%s",name.c_str());
    char hruntitle[400];
    sprintf(hruntitle,"Number of bad modules %s",name.c_str());
    m_badmodrun[name] = m_rhm.makeTProfile(hrunname,hruntitle,m_LSfrac*m_maxLS,0,m_maxLS*262144);
  }

}


SiPixelQualityHistory::~SiPixelQualityHistory()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiPixelQualityHistory::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  edm::LogInfo("EventProcessing") << "event being processed";

  for(std::vector<edm::ParameterSet>::const_iterator ps=m_monitoredspq.begin();ps!=m_monitoredspq.end();++ps) {

    std::string name = ps->getParameter<std::string>("name");
    std::string label = ps->getParameter<std::string>("spqLabel");



    edm::ESHandle<SiPixelQuality> spq;
    iSetup.get<SiPixelQualityRcd>().get(label,spq);

    int nbad=0;

    if(m_mode==Summary) {

      //      nbad = spq->BadModuleNumber();

    }
    else {

      std::vector<SiPixelQuality::disabledModuleType> bads = spq->getBadComponentList();

      LogDebug("BadComponents") << bads.size() << " bad components found";

      for(std::vector<SiPixelQuality::disabledModuleType>::const_iterator bc=bads.begin();bc!=bads.end();++bc) {

	if(m_mode==Module) {
	  if(spq->IsModuleBad(bc->DetID)) ++nbad;
	  //	  if(bc->errorType==0) ++nbad;
	}
	else if(m_mode == ROC) {
	  for(int roc=1;roc<2*2*2*2*2*2*2+1;roc*=2) {
	    if((bc->BadRocs & roc)>0) ++nbad;
	  }
	}
      }
    }
    if(m_badmodrun.find(name)!=m_badmodrun.end() && m_badmodrun[name] && *m_badmodrun[name]) {
      (*m_badmodrun[name])->Fill(iEvent.orbitNumber(),nbad);
    }

  }
}

void
SiPixelQualityHistory::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  m_rhm.beginRun(iRun);

  // loop on all the SiPixelQuality objects to be monitored

  for(std::vector<edm::ParameterSet>::const_iterator ps=m_monitoredspq.begin();ps!=m_monitoredspq.end();++ps) {

    std::string name = ps->getParameter<std::string>("name");
    std::string label = ps->getParameter<std::string>("spqLabel");

    if(m_badmodrun.find(name)!=m_badmodrun.end()) {
      if(m_badmodrun[name] && *m_badmodrun[name]) {
	(*m_badmodrun[name])->SetCanExtend(TH1::kXaxis);
	(*m_badmodrun[name])->GetXaxis()->SetTitle("time [Orb#]"); (*m_badmodrun[name])->GetYaxis()->SetTitle("bad components");
      }
    }

    if(m_run) {

      edm::ESHandle<SiPixelQuality> spq;
      iSetup.get<SiPixelQualityRcd>().get(label,spq);

      int nbad=0;

      if(m_mode==Summary) {

	//	nbad = spq->BadModuleNumber();

      }
      else {

	std::vector<SiPixelQuality::disabledModuleType> bads = spq->getBadComponentList();

	LogDebug("BadComponents") << bads.size() << " bad components found";

	for(std::vector<SiPixelQuality::disabledModuleType>::const_iterator bc=bads.begin();bc!=bads.end();++bc) {

	  if(m_mode==Module) {
	    if(spq->IsModuleBad(bc->DetID)) ++nbad;
	    //	  if(bc->errorType==0) ++nbad;
	  }
	  else if(m_mode == ROC) {
	    for(int roc=1;roc<2*2*2*2*2*2*2+1;roc*=2) {
	      if((bc->BadRocs & roc)>0) ++nbad;
	    }
	  }
	}
      }
      char runname[100];
      sprintf(runname,"%d",iRun.run());
      LogDebug("AnalyzedRun") << name << " " << runname << " " << nbad;
      m_history[name]->Fill(runname,nbad);
    }
  }

}

// ------------ method called once each job just before starting event loop  ------------
void
SiPixelQualityHistory::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void
SiPixelQualityHistory::endJob()
{
  /*
  for(std::vector<edm::ParameterSet>::const_iterator ps=m_monitoredspq.begin();ps!=m_monitoredspq.end();++ps) {

    std::string name = ps->getParameter<std::string>("name");
    m_history[name]->Write();

  }
  */
}


//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelQualityHistory);
