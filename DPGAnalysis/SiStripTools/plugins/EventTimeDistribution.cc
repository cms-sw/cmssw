// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      EventTimeDistribution
// 
/**\class EventTimeDistribution EventTimeDistribution.cc DPGAnalysis/SiStripTools/plugins/EventTimeDistribution.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jul 19 11:56:00 CEST 2009
//
//


// system include files
#include <memory>

// user include files
#include <string>

#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
//
// class decleration
//

class EventTimeDistribution : public edm::EDAnalyzer {
 public:
    explicit EventTimeDistribution(const edm::ParameterSet&);
    ~EventTimeDistribution();


   private:
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  const edm::InputTag _historyProduct;
  const edm::InputTag _apvphasecoll;
  const std::string _phasepart;
  const bool _wantdbxvsbxincycle;
  const bool _wantdbxvsbx;
  const bool _wantbxincyclevsbx;
  const bool _wantorbitvsbxincycle;
  unsigned int _nevents;
  const unsigned int m_maxLS;

  RunHistogramManager _rhm;

  TH1F** _dbx;
  TH1F** _bx;
  TH1F** _bxincycle;
  TH1F** _orbit;
  TH2F** _dbxvsbxincycle;
  TH2F** _dbxvsbx;
  TH2F** _bxincyclevsbx;
  TH2F** _orbitvsbxincycle;

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
EventTimeDistribution::EventTimeDistribution(const edm::ParameterSet& iConfig):
  _historyProduct(iConfig.getParameter<edm::InputTag>("historyProduct")),
  _apvphasecoll(iConfig.getParameter<edm::InputTag>("apvPhaseCollection")),
  _phasepart(iConfig.getUntrackedParameter<std::string>("phasePartition","None")),
  _wantdbxvsbxincycle(iConfig.getUntrackedParameter<bool>("wantDBXvsBXincycle",false)),
  _wantdbxvsbx(iConfig.getUntrackedParameter<bool>("wantDBXvsBX",false)),
  _wantbxincyclevsbx(iConfig.getUntrackedParameter<bool>("wantBXincyclevsBX",false)),
  _wantorbitvsbxincycle(iConfig.getUntrackedParameter<bool>("wantOrbitvsBXincycle",false)),
  _nevents(0),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",100)),
  _rhm(),
  _dbxvsbxincycle(0),   _dbxvsbx(0),   _bxincyclevsbx(0),   _orbitvsbxincycle(0)
{
   //now do what ever initialization is needed

  _dbx = _rhm.makeTH1F("dbx","dbx",1000,-0.5,999.5);
  _bx = _rhm.makeTH1F("bx","BX number",3564,-0.5,3563.5);
  _bxincycle = _rhm.makeTH1F("bxcycle","bxcycle",70,-0.5,69.5);
  _orbit = _rhm.makeTH1F("orbit","orbit",4*m_maxLS,0,m_maxLS*262144);
  if(_wantdbxvsbxincycle) _dbxvsbxincycle = _rhm.makeTH2F("dbxvsbxincycle","dbxvsbxincycle",70,-0.5,69.5,1000,-0.5,999.5);
  if(_wantdbxvsbx) _dbxvsbx = _rhm.makeTH2F("dbxvsbx","dbxvsbx",3564,-0.5,3563.5,1000,-0.5,999.5);
  if(_wantbxincyclevsbx) _bxincyclevsbx = _rhm.makeTH2F("bxincyclevsbx","bxincyclevsbx",3564,-0.5,3563.5,70,-0.5,69.5);
  if(_wantorbitvsbxincycle) _orbitvsbxincycle = _rhm.makeTH2F("orbitvsbxincycle","orbitvsbxincycle",70,-0.5,69.5,m_maxLS,0,m_maxLS*262144);

  edm::LogInfo("UsedAPVCyclePhaseCollection") << " APVCyclePhaseCollection " << _apvphasecoll << " used";

}


EventTimeDistribution::~EventTimeDistribution()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EventTimeDistribution::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nevents++;

   edm::Handle<EventWithHistory> he;
   iEvent.getByLabel(_historyProduct,he);

   edm::Handle<APVCyclePhaseCollection> apvphase;
   iEvent.getByLabel(_apvphasecoll,apvphase);

   long long tbx = he->absoluteBX();
   if(apvphase.isValid() && !apvphase.failedToGet()) {
     const int thephase = apvphase->getPhase(_phasepart); 
     if(thephase!=APVCyclePhaseCollection::invalid &&
	thephase!=APVCyclePhaseCollection::multiphase &&
	thephase!=APVCyclePhaseCollection::nopartition)
       tbx -= thephase;
   }



   // improve the matchin between default and actual partitions
   
   (*_dbx)->Fill(he->deltaBX());
   (*_bx)->Fill(iEvent.bunchCrossing());
   (*_bxincycle)->Fill(tbx%70);
   (*_orbit)->Fill(iEvent.orbitNumber());
   if(_dbxvsbxincycle && *_dbxvsbxincycle) (*_dbxvsbxincycle)->Fill(tbx%70,he->deltaBX());
   if(_dbxvsbx && *_dbxvsbx) (*_dbxvsbx)->Fill(iEvent.bunchCrossing(),he->deltaBX());
   if(_bxincyclevsbx && *_bxincyclevsbx) (*_bxincyclevsbx)->Fill(iEvent.bunchCrossing(),tbx%70);
   if(_orbitvsbxincycle && *_orbitvsbxincycle) (*_orbitvsbxincycle)->Fill(tbx%70,iEvent.orbitNumber());


}

void 
EventTimeDistribution::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{

  _rhm.beginRun(iRun);
  if(*_dbx) {    (*_dbx)->GetXaxis()->SetTitle("#DeltaBX"); }

  if(*_bx) { (*_bx)->GetXaxis()->SetTitle("BX");  }

  if(*_bxincycle) {  (*_bxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); }

  if(*_orbit) {
    (*_orbit)->SetBit(TH1::kCanRebin);
    (*_orbit)->GetXaxis()->SetTitle("time [Orb#]"); 
  }

  if(_dbxvsbxincycle && *_dbxvsbxincycle) {
    (*_dbxvsbxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); (*_dbxvsbxincycle)->GetYaxis()->SetTitle("#DeltaBX"); 
  }

  if(_dbxvsbx && *_dbxvsbx) { (*_dbxvsbx)->GetXaxis()->SetTitle("BX"); (*_dbxvsbx)->GetYaxis()->SetTitle("#DeltaBX"); }

  if(_bxincyclevsbx && *_bxincyclevsbx) {
    (*_bxincyclevsbx)->GetXaxis()->SetTitle("BX"); (*_bxincyclevsbx)->GetYaxis()->SetTitle("Event BX mod(70)");
  }

  if(_orbitvsbxincycle && *_orbitvsbxincycle) {
    (*_orbitvsbxincycle)->SetBit(TH1::kCanRebin);
    (*_orbitvsbxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); (*_orbitvsbxincycle)->GetYaxis()->SetTitle("time [Orb#]"); 
  }
}

void 
EventTimeDistribution::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
EventTimeDistribution::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventTimeDistribution::endJob() {

  edm::LogInfo("EndOfJob") << _nevents << " analyzed events";

}


//define this as a plug-in
DEFINE_FWK_MODULE(EventTimeDistribution);
