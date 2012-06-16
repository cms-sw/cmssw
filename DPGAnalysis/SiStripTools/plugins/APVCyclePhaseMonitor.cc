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
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  const edm::InputTag _apvphasecollection;
  std::vector<std::string> _selectedparts;
  std::vector<std::string> _selectedvectorparts;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  std::map<std::string,TH1F*> _hphases;
  std::map<std::string,TH1F*> _hselectedphases;
  std::map<std::string,TH1F*> _hselectedphasesvector;
  std::map<std::string,TH1F*> _hselectedphasessize;
  std::map<std::string,TProfile*> _hphasevsorbit;
  std::map<std::string,TProfile*> _hselectedphasevsorbit;
  std::map<std::string,TProfile*> _hselectedphasevectorvsorbit;
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
  _selectedparts(iConfig.getUntrackedParameter<std::vector<std::string> >("selectedPartitions",std::vector<std::string>())),
  _selectedvectorparts(iConfig.getUntrackedParameter<std::vector<std::string> >("selectedVectorPartitions",std::vector<std::string>())),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",125)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",16)),
  _hphases(),_hselectedphases(),_hselectedphasesvector(),_hselectedphasessize(),
  _hphasevsorbit(),_hselectedphasevsorbit(),_hselectedphasevectorvsorbit(),
  _nevents(0)
{
   //now do what ever initialization is needed

  edm::LogInfo("UsedAPVCyclePhaseCollection") << " APVCyclePhaseCollection " << _apvphasecollection << " used";

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
       _hphasevsorbit[phase->first] = subrun.make<TProfile>(hname,hname,m_LSfrac*m_maxLS,0,m_maxLS*262144);
       _hphasevsorbit[phase->first]->SetBit(TH1::kCanRebin);
       _hphasevsorbit[phase->first]->GetXaxis()->SetTitle("time [orbit#]"); _hphasevsorbit[phase->first]->GetYaxis()->SetTitle("Phase");
       
     }
     _hphases[phase->first]->Fill(phase->second);
     _hphasevsorbit[phase->first]->Fill(iEvent.orbitNumber(),phase->second);
   }

   // selected partitions
   
   for(std::vector<std::string>::const_iterator part=_selectedparts.begin();part!=_selectedparts.end();++part) {
     if(_hselectedphases.find(*part)!=_hselectedphases.end()) {
       _hselectedphases[*part]->Fill(apvphases->getPhase(*part));
     }
     if(_hselectedphasevsorbit.find(*part)!=_hselectedphasevsorbit.end()) {
       _hselectedphasevsorbit[*part]->Fill(iEvent.orbitNumber(),apvphases->getPhase(*part));
     }
   }

   for(std::vector<std::string>::const_iterator part=_selectedvectorparts.begin();
       part!=_selectedvectorparts.end();++part) {

     const std::vector<int> phases = apvphases->getPhases(*part);

      if(_hselectedphasessize.find(*part)!=_hselectedphasessize.end()) {
	_hselectedphasessize[*part]->Fill(phases.size());
      }

     for(std::vector<int>::const_iterator phase=phases.begin();phase!=phases.end();++phase) {
       if(_hselectedphasesvector.find(*part)!=_hselectedphasesvector.end()) {
	 _hselectedphasesvector[*part]->Fill(*phase);
       }
       if(_hselectedphasevectorvsorbit.find(*part)!=_hselectedphasevectorvsorbit.end()) {
	 _hselectedphasevectorvsorbit[*part]->Fill(iEvent.orbitNumber(),*phase);
       }
     }

   }
}

void 
APVCyclePhaseMonitor::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
  
  _hphases.clear();
  _hselectedphases.clear();
  _hselectedphasesvector.clear();
  _hselectedphasessize.clear();
  _hphasevsorbit.clear();
  _hselectedphasevsorbit.clear();
  _hselectedphasevectorvsorbit.clear();

  edm::Service<TFileService> tfserv;

  char dirname[300];
  sprintf(dirname,"run_%d",iRun.run());
  TFileDirectory subrun = tfserv->mkdir(dirname);
  
  for(std::vector<std::string>::const_iterator part=_selectedparts.begin();part!=_selectedparts.end();++part) {

    char hname[300];
    
    sprintf(hname,"selected_phase_%s",part->c_str());
    edm::LogInfo("SelectedTH1FBeingBooked") << "TH1F " << hname << " being booked" ;
    _hselectedphases[*part] = subrun.make<TH1F>(hname,hname,70,-0.5,69.5);
    _hselectedphases[*part]->GetXaxis()->SetTitle("BX mod 70"); _hselectedphases[*part]->GetYaxis()->SetTitle("Events");
    
    sprintf(hname,"selected_phasevsorbit_%s",part->c_str());
    edm::LogInfo("SelectedTProfileBeingBooked") << "TProfile " << hname << " being booked" ;
    _hselectedphasevsorbit[*part] = subrun.make<TProfile>(hname,hname,m_LSfrac*m_maxLS,0,m_maxLS*262144);
    _hselectedphasevsorbit[*part]->SetBit(TH1::kCanRebin);
    _hselectedphasevsorbit[*part]->GetXaxis()->SetTitle("time [orbit#]"); 
    _hselectedphasevsorbit[*part]->GetYaxis()->SetTitle("Phase");
  }

  for(std::vector<std::string>::const_iterator part=_selectedvectorparts.begin();
      part!=_selectedvectorparts.end();++part) {

    char hname[300];
    
    sprintf(hname,"selected_phase_vector_%s",part->c_str());
    edm::LogInfo("SelectedVectTH1FBeingBooked") << "TH1F " << hname << " being booked" ;
    _hselectedphasesvector[*part] = subrun.make<TH1F>(hname,hname,70,-0.5,69.5);
    _hselectedphasesvector[*part]->GetXaxis()->SetTitle("BX mod 70"); 
    _hselectedphasesvector[*part]->GetYaxis()->SetTitle("Events");
    
    sprintf(hname,"selected_phase_size_%s",part->c_str());
    edm::LogInfo("SelectedVectSizeTH1FBeingBooked") << "TH1F " << hname << " being booked" ;
    _hselectedphasessize[*part] = subrun.make<TH1F>(hname,hname,10,-0.5,9.5);
    _hselectedphasessize[*part]->GetXaxis()->SetTitle("Number of Phases"); 
    _hselectedphasessize[*part]->GetYaxis()->SetTitle("Events");
    
    sprintf(hname,"selected_phasevectorvsorbit_%s",part->c_str());
    edm::LogInfo("SelectedVectTProfileBeingBooked") << "TProfile " << hname << " being booked" ;
    _hselectedphasevectorvsorbit[*part] = subrun.make<TProfile>(hname,hname,m_LSfrac*m_maxLS,0,m_maxLS*262144);
    _hselectedphasevectorvsorbit[*part]->SetBit(TH1::kCanRebin);
    _hselectedphasevectorvsorbit[*part]->GetXaxis()->SetTitle("time [orbit#]"); 
    _hselectedphasevectorvsorbit[*part]->GetYaxis()->SetTitle("Phase");
  }
}

void 
APVCyclePhaseMonitor::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
APVCyclePhaseMonitor::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
APVCyclePhaseMonitor::endJob() {

  edm::LogInfo("EndOfJob") << _nevents << " analyzed events";

}


//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseMonitor);
