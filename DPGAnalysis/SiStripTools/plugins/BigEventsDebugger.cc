// -*- C++ -*-
//
// Package:    BigEventsDebugger
// Class:      BigEventsDebugger
// 
/**\class BigEventsDebugger BigEventsDebugger.cc myTKAnalyses/BigEventsDebugger/src/BigEventsDebugger.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Sun Nov 16 16:04:44 CET 2008
// $Id: BigEventsDebugger.cc,v 1.3 2013/02/27 19:49:46 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DPGAnalysis/SiStripTools/interface/DigiCollectionProfiler.h"

//
// class decleration
//

template <class T>
class BigEventsDebugger : public edm::EDAnalyzer {
   public:
      explicit BigEventsDebugger(const edm::ParameterSet&);
      ~BigEventsDebugger();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      // ----------member data ---------------------------

  DigiCollectionProfiler<T> m_digiprofiler;
  edm::InputTag m_collection;
  bool m_singleevents;
  bool m_folded;
  bool m_want1dHisto;
  bool m_wantProfile;
  bool m_want2dHisto;

  std::vector<std::string> m_labels;
  std::vector<TH1F*> m_hist;
  std::vector<TProfile*> m_hprof;
  std::vector<TH2F*> m_hist2d;

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
template <class T>
BigEventsDebugger<T>::BigEventsDebugger(const edm::ParameterSet& iConfig):
  m_digiprofiler(iConfig),
  m_collection(iConfig.getParameter<edm::InputTag>("collection")),
  m_singleevents(iConfig.getParameter<bool>("singleEvents")),
  m_folded(iConfig.getUntrackedParameter<bool>("foldedStrips",false)),
  m_want1dHisto(iConfig.getUntrackedParameter<bool>("want1dHisto",true)),
  m_wantProfile(iConfig.getUntrackedParameter<bool>("wantProfile",true)),
  m_want2dHisto(iConfig.getUntrackedParameter<bool>("want2dHisto",false))

{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet> >("selections");
  
  for(std::vector<edm::ParameterSet>::const_iterator selconfig=selconfigs.begin();selconfig!=selconfigs.end();++selconfig) {
    m_labels.push_back(selconfig->getParameter<std::string>("label"));
  }


  edm::Service<TFileService> tfserv;

  if(!m_singleevents) {
    char dirname[500];
    sprintf(dirname,"Summary");
    TFileDirectory subd = tfserv->mkdir(dirname);
    
    //book histos
    
    unsigned int nbins =768;
    if(m_folded) nbins=256;
    
    for(std::vector<std::string>::const_iterator label=m_labels.begin(); label!=m_labels.end(); ++label) {
      if(m_want1dHisto) {
	std::string hname = *label + "hist";
	std::string htitle = *label + " occupancy";
	m_hist.push_back(subd.make<TH1F>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5));
      }
      if(m_wantProfile) {
	std::string hname = *label + "prof";
	std::string htitle = *label + " charge profile";
	m_hprof.push_back(subd.make<TProfile>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5));
      }
      if(m_want2dHisto) {
	std::string hname = *label + "hist2d";
	std::string htitle = *label + " charge distribution";
	m_hist2d.push_back(subd.make<TH2F>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5,257,-0.5,256.5));
      }
    }
  }
}

template <class T>
BigEventsDebugger<T>::~BigEventsDebugger()
{

 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
template <class T>
void
BigEventsDebugger<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Service<TFileService> tfserv;

   // create a folder for each event

   if(m_singleevents) {

     m_hist.clear();     m_hprof.clear();     m_hist2d.clear();
     
     char dirname[500];
     sprintf(dirname,"event_%u_%u",iEvent.run(),iEvent.id().event());
     TFileDirectory subd = tfserv->mkdir(dirname);
     
     //book histos
     
     unsigned int nbins =768;
     if(m_folded) nbins=256;
     
     for(std::vector<std::string>::const_iterator label=m_labels.begin(); label!=m_labels.end(); ++label) {
       if(m_want1dHisto) {
	 std::string hname = *label + "hist";
	 std::string htitle = *label + " occupancy";
	 m_hist.push_back(subd.make<TH1F>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5));
       }
       if(m_wantProfile) {
	 std::string hname = *label + "prof";
	 std::string htitle = *label + " charge profile";
	 m_hprof.push_back(subd.make<TProfile>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5));
       }
       if(m_want2dHisto) {
	 std::string hname = *label + "hist2d";
	 std::string htitle = *label + " charge distribution";
	 m_hist2d.push_back(subd.make<TH2F>(hname.c_str(),htitle.c_str(),nbins,-0.5,nbins-0.5,257,-0.5,256.5));
       }
     }
     
   }

   //analyze event

   Handle<T> digis;
   iEvent.getByLabel(m_collection,digis);
   m_digiprofiler.fill(digis,m_hist,m_hprof,m_hist2d);

}


// ------------ method called once each job just before starting event loop  ------------
template <class T>
void 
BigEventsDebugger<T>::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <class T>
void 
BigEventsDebugger<T>::endJob() {
}

typedef BigEventsDebugger<edmNew::DetSetVector<SiStripCluster> > ClusterBigEventsDebugger;
typedef BigEventsDebugger<edm::DetSetVector<SiStripDigi> > DigiBigEventsDebugger;

//define this as a plug-in
DEFINE_FWK_MODULE(ClusterBigEventsDebugger);
DEFINE_FWK_MODULE(DigiBigEventsDebugger);
