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
#include <vector>

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
      virtual void beginJob() override ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------

  edm::EDGetTokenT<EventWithHistory> _historyProductToken;
  edm::EDGetTokenT<APVCyclePhaseCollection> _apvphasecollToken;
  const std::string _phasepart;
  const bool _wantdbxvsbxincycle;
  const bool _wantdbxvsbx;
  const bool _wantbxincyclevsbx;
  const bool _wantorbitvsbxincycle;
  unsigned int _nevents;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  const bool m_ewhdepthHisto;



  RunHistogramManager _rhm;

  TH1F** _dbx;
  std::vector<TH1F**> m_dbxhistos;
  std::vector<std::pair<unsigned int,unsigned int> > m_dbxindices;
  TH1F** _bx;
  TH1F** _bxincycle;
  TH1F** _orbit;
  TH2F** _dbxvsbxincycle;
  TH2F** _dbxvsbx;
  TH2F** _bxincyclevsbx;
  TH2F** _orbitvsbxincycle;
  TH1F** m_ewhdepth;

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
  _historyProductToken(consumes<EventWithHistory>(iConfig.getParameter<edm::InputTag>("historyProduct"))),
  _apvphasecollToken(consumes<APVCyclePhaseCollection>(iConfig.getParameter<edm::InputTag>("apvPhaseCollection"))),
  _phasepart(iConfig.getUntrackedParameter<std::string>("phasePartition","None")),
  _wantdbxvsbxincycle(iConfig.getUntrackedParameter<bool>("wantDBXvsBXincycle",false)),
  _wantdbxvsbx(iConfig.getUntrackedParameter<bool>("wantDBXvsBX",false)),
  _wantbxincyclevsbx(iConfig.getUntrackedParameter<bool>("wantBXincyclevsBX",false)),
  _wantorbitvsbxincycle(iConfig.getUntrackedParameter<bool>("wantOrbitvsBXincycle",false)),
  _nevents(0),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",100)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",4)),
  m_ewhdepthHisto(iConfig.getUntrackedParameter<bool>("wantEWHDepthHisto",false)),
  _rhm(consumesCollector()),
  _dbxvsbxincycle(0),   _dbxvsbx(0),   _bxincyclevsbx(0),   _orbitvsbxincycle(0), m_ewhdepth(0)
{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> dbxhistoparams(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("dbxHistosParams",std::vector<edm::ParameterSet>()));

  for(std::vector<edm::ParameterSet>::const_iterator params=dbxhistoparams.begin();params!=dbxhistoparams.end();++params) {
    m_dbxindices.push_back(std::pair<unsigned int,unsigned int>(params->getParameter<unsigned int>("firstEvent"),params->getParameter<unsigned int>("secondEvent")));
    char hname[300];
    sprintf(hname,"dbx_%d_%d",params->getParameter<unsigned int>("firstEvent"),params->getParameter<unsigned int>("secondEvent"));
    char htitle[300];
    sprintf(htitle,"dbx(%d,%d)",params->getParameter<unsigned int>("firstEvent"),params->getParameter<unsigned int>("secondEvent"));

    m_dbxhistos.push_back(_rhm.makeTH1F(hname,htitle,params->getParameter<int>("nbins"),params->getParameter<double>("min"),
					params->getParameter<double>("max")));
    LogDebug("DBXHistoPreBooking") << "Booked DBX histo named " << hname << " untitled " << htitle;
  }



  _dbx = _rhm.makeTH1F("dbx","dbx",1000,-0.5,999.5);
  _bx = _rhm.makeTH1F("bx","BX number",3564,-0.5,3563.5);
  _bxincycle = _rhm.makeTH1F("bxcycle","bxcycle",70,-0.5,69.5);
  _orbit = _rhm.makeTH1F("orbit","orbit",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  if(_wantdbxvsbxincycle) _dbxvsbxincycle = _rhm.makeTH2F("dbxvsbxincycle","dbxvsbxincycle",70,-0.5,69.5,1000,-0.5,999.5);
  if(_wantdbxvsbx) _dbxvsbx = _rhm.makeTH2F("dbxvsbx","dbxvsbx",3564,-0.5,3563.5,1000,-0.5,999.5);
  if(_wantbxincyclevsbx) _bxincyclevsbx = _rhm.makeTH2F("bxincyclevsbx","bxincyclevsbx",3564,-0.5,3563.5,70,-0.5,69.5);
  if(_wantorbitvsbxincycle) _orbitvsbxincycle = _rhm.makeTH2F("orbitvsbxincycle","orbitvsbxincycle",70,-0.5,69.5,m_maxLS,0,m_maxLS*262144);
  if(m_ewhdepthHisto) m_ewhdepth = _rhm.makeTH1F("ewhdepth","EventWithHistory Depth",11,-0.5,10.5);

  edm::LogInfo("UsedAPVCyclePhaseCollection") << " APVCyclePhaseCollection " << iConfig.getParameter<edm::InputTag>("apvPhaseCollection") << " used";

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
   iEvent.getByToken(_historyProductToken,he);

   // improve the matchin between default and actual partitions

   (*_dbx)->Fill(he->deltaBX());
   std::vector<std::pair<unsigned int,unsigned int> >::const_iterator indices=m_dbxindices.begin();
   for(std::vector<TH1F**>::const_iterator dbxhist=m_dbxhistos.begin();dbxhist!=m_dbxhistos.end();++dbxhist,++indices) {
     (*(*dbxhist))->Fill(he->deltaBX(indices->first,indices->second));
   }

   (*_bx)->Fill(iEvent.bunchCrossing());
   (*_orbit)->Fill(iEvent.orbitNumber());
   if(_dbxvsbx && *_dbxvsbx) (*_dbxvsbx)->Fill(iEvent.bunchCrossing(),he->deltaBX());
   if(m_ewhdepth && *m_ewhdepth) (*m_ewhdepth)->Fill(he->depth());

   edm::Handle<APVCyclePhaseCollection> apvphase;
   iEvent.getByToken(_apvphasecollToken,apvphase);

   long long tbx = he->absoluteBX();
   if(apvphase.isValid() && !apvphase.failedToGet()) {
     const int thephase = apvphase->getPhase(_phasepart);
     if(thephase!=APVCyclePhaseCollection::invalid &&
	thephase!=APVCyclePhaseCollection::multiphase &&
	thephase!=APVCyclePhaseCollection::nopartition) {

       tbx -= thephase;
       (*_bxincycle)->Fill(tbx%70);
       if(_dbxvsbxincycle && *_dbxvsbxincycle) (*_dbxvsbxincycle)->Fill(tbx%70,he->deltaBX());
       if(_bxincyclevsbx && *_bxincyclevsbx) (*_bxincyclevsbx)->Fill(iEvent.bunchCrossing(),tbx%70);
       if(_orbitvsbxincycle && *_orbitvsbxincycle) (*_orbitvsbxincycle)->Fill(tbx%70,iEvent.orbitNumber());

     }
     else {
       LogDebug("InvalidPhase") << "Invalid APVCyclePhase value : " << _phasepart << " " << thephase;
     }
   }
}

void
EventTimeDistribution::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{

  _rhm.beginRun(iRun);

  if(*_dbx) {    (*_dbx)->GetXaxis()->SetTitle("#DeltaBX"); }

  LogDebug("NomberOfHistos") << m_dbxhistos.size();
  for(std::vector<TH1F**>::const_iterator dbxhist=m_dbxhistos.begin();dbxhist!=m_dbxhistos.end();++dbxhist) {
    LogDebug("HistoPointer") << *dbxhist;
    if(*(*dbxhist)) { (*(*dbxhist))->GetXaxis()->SetTitle("#DeltaBX"); }
  }
  LogDebug("LabelDone") << "all labels set";

  if(*_bx) { (*_bx)->GetXaxis()->SetTitle("BX");  }

  if(*_bxincycle) {  (*_bxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); }

  if(*_orbit) {
    (*_orbit)->SetCanExtend(TH1::kXaxis);
    (*_orbit)->GetXaxis()->SetTitle("time [Orb#]");
  }

  LogDebug("StdPlotsDone") << "all labels in std plots set";

  if(_dbxvsbxincycle && *_dbxvsbxincycle) {
    (*_dbxvsbxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); (*_dbxvsbxincycle)->GetYaxis()->SetTitle("#DeltaBX");
  }

  if(_dbxvsbx && *_dbxvsbx) { (*_dbxvsbx)->GetXaxis()->SetTitle("BX"); (*_dbxvsbx)->GetYaxis()->SetTitle("#DeltaBX"); }

  if(_bxincyclevsbx && *_bxincyclevsbx) {
    (*_bxincyclevsbx)->GetXaxis()->SetTitle("BX"); (*_bxincyclevsbx)->GetYaxis()->SetTitle("Event BX mod(70)");
  }

  if(_orbitvsbxincycle && *_orbitvsbxincycle) {
    (*_orbitvsbxincycle)->SetCanExtend(TH1::kYaxis);
    (*_orbitvsbxincycle)->GetXaxis()->SetTitle("Event BX mod(70)"); (*_orbitvsbxincycle)->GetYaxis()->SetTitle("time [Orb#]");
  }

  if(m_ewhdepth && *m_ewhdepth) {
    (*m_ewhdepth)->GetXaxis()->SetTitle("Depth");
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
