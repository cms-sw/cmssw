// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      APVCyclePhaseMonitor
// 
/**\class APVCyclePhaseMonitor APVCyclePhaseMonitor.cc DPGAnalysis/SiStripTools/plugins/APVCyclePhaseMonitor.cc

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
#include "TH1F.h"
#include "TProfile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

//
// class decleration
//

class APVCyclePhaseMonitor : public edm::EDAnalyzer {
 public:
    explicit APVCyclePhaseMonitor(const edm::ParameterSet&);
    ~APVCyclePhaseMonitor();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  const edm::InputTag _apvphasecollection;
  std::map<std::string,TH1F*> _hphases;
  std::map<std::string,TProfile*> _hphasevsorbit;
  unsigned int _nevents;
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
APVCyclePhaseMonitor::APVCyclePhaseMonitor(const edm::ParameterSet& iConfig):
  _apvphasecollection(iConfig.getParameter<edm::InputTag>("apvCyclePhaseCollection")),
  _hphases(),_hphasevsorbit(),
  _nevents(0)
{
   //now do what ever initialization is needed

}


APVCyclePhaseMonitor::~APVCyclePhaseMonitor()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
APVCyclePhaseMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nevents++;

   edm::Handle<APVCyclePhaseCollection> apvphases;
   iEvent.getByLabel(_apvphasecollection,apvphases);

   // improve the matchin between default and actual partitions
   
   edm::Service<TFileService> tfserv;

   for(std::map<std::string,int>::const_iterator phase = apvphases->get().begin(); phase != apvphases->get().end(); ++phase) {

     if(_hphases.find(phase->first)==_hphases.end()) {
       char dirname[300];
       sprintf(dirname,"run_%d",iEvent.run());
       TFileDirectory subrun = tfserv->mkdir(dirname);

       char hname[300];

       sprintf(hname,"phase_%s",phase->first.c_str());
       edm::LogInfo("TH1FBeingBooked") << "TH1F " << hname << " being booked" ;
       _hphases[phase->first] = subrun.make<TH1F>(hname,hname,70,-0.5,69.5);
       _hphases[phase->first]->GetXaxis()->SetTitle("BX mod 70"); _hphases[phase->first]->GetYaxis()->SetTitle("Events");

       sprintf(hname,"phasevsorbit_%s",phase->first.c_str());
       edm::LogInfo("TProfileBeingBooked") << "TProfile " << hname << " being booked" ;
       _hphasevsorbit[phase->first] = subrun.make<TProfile>(hname,hname,1800,0.,1800*11223);
       _hphasevsorbit[phase->first]->SetBit(TH1::kCanRebin);
       _hphasevsorbit[phase->first]->GetXaxis()->SetTitle("time [orbit#]"); _hphasevsorbit[phase->first]->GetYaxis()->SetTitle("Phase");
       
     }
     _hphases[phase->first]->Fill(phase->second);
     _hphasevsorbit[phase->first]->Fill(iEvent.orbitNumber(),phase->second);
   }
   
}

void 
APVCyclePhaseMonitor::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
  
  _hphases.clear();
  _hphasevsorbit.clear();
  
  /*
  for(std::vector<std::string>::const_iterator part = _defpartnames.begin(); part != _defpartnames.end() ; ++part) {
    sprintf(hname,"phase_%s",part->c_str());
    _hphases[*part] = subrun.make<TH1F>(hname,hname,70,-0.5,69.5);
    sprintf(hname,"phasevsorbit_%s",part->c_str());
    _hphasevsorbit[*part] = subrun.make<TProfile>(hname,hname,1800,0.,1800*11223);
    _hphasevsorbit[*part]->SetBit(TH1::kCanRebin);
  }
  */
  
}

void 
APVCyclePhaseMonitor::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
APVCyclePhaseMonitor::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
APVCyclePhaseMonitor::endJob() {

  edm::LogInfo("EndOfJob") << _nevents << " analyzed events";

}


//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseMonitor);
