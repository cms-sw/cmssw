// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      APVShotsAnalyzer
// 
/**\class APVShotsAnalyzer APVShotsAnalyzer.cc DPGAnalysis/SiStripTools/plugins/APVShotsAnalyzer.cc

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
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DPGAnalysis/SiStripTools/interface/APVShotFinder.h"
#include "DPGAnalysis/SiStripTools/interface/APVShot.h"

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
//***************************************************

//
// class decleration
//

class APVShotsAnalyzer : public edm::EDAnalyzer {
 public:
    explicit APVShotsAnalyzer(const edm::ParameterSet&);
    ~APVShotsAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  edm::InputTag _digicollection;
  bool _zs;
  std::string _suffix;
  int _nevents;

  TH1F* _nShots;
  TProfile* _nShotsVsTime;
  TH1F* _whichAPV;
  TH1F* _stripMult;
  TH1F* _median;
  TH1F* _subDetector;

  TkHistoMap *tkhisto,*tkhisto2; 
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
APVShotsAnalyzer::APVShotsAnalyzer(const edm::ParameterSet& iConfig):
  _digicollection(iConfig.getParameter<edm::InputTag>("digiCollection")),
  _zs(iConfig.getUntrackedParameter<bool>("zeroSuppressed",true)),
  _suffix(iConfig.getParameter<std::string>("mapSuffix")),
  _nevents(0)
{
   //now do what ever initialization is needed

  if(!_zs) _suffix += "_notZS";

 edm::Service<TFileService> tfserv;

 _nShots = tfserv->make<TH1F>("nShots","Number of Shots per event",200,-0.5,199.5);
 _nShots->GetXaxis()->SetTitle("Shots");  _nShots->GetYaxis()->SetTitle("Events"); 
 _nShots->StatOverflows(kTRUE);

 _nShotsVsTime = tfserv->make<TProfile>("nShotsVsTime","Mean number of shots vs orbit number",3600,0.5,3600*11223+0.5);
 _nShotsVsTime->GetXaxis()->SetTitle("Orbit");  _nShotsVsTime->GetYaxis()->SetTitle("Number of Shots");
 _nShotsVsTime->SetBit(TH1::kCanRebin);

 _whichAPV = tfserv->make<TH1F>("whichAPV","APV with shots",6,-0.5,5.5);
 _whichAPV->GetXaxis()->SetTitle("APV");  _whichAPV->GetYaxis()->SetTitle("Shots"); 

 _stripMult = tfserv->make<TH1F>("stripMultiplicity","Shot Strip Multiplicity",129,-0.5,128.5);
 _stripMult->GetXaxis()->SetTitle("Number of Strips");  _stripMult->GetYaxis()->SetTitle("Shots");

 _median = tfserv->make<TH1F>("median","APV Shot charge median",256,-0.5,255.5);
 _median->GetXaxis()->SetTitle("Charge [ADC]");  _median->GetYaxis()->SetTitle("Shots");

 _subDetector = tfserv->make<TH1F>("subDets","SubDetector Shot distribution",10,-0.5,9.5);
 _subDetector->GetYaxis()->SetTitle("Shots");


 tkhisto      =new TkHistoMap("ShotMultiplicity","ShotMultiplicity",-1); 
 tkhisto2      =new TkHistoMap("StripMultiplicity","StripMultiplicity",-1); 
}


APVShotsAnalyzer::~APVShotsAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
APVShotsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nevents++;

   Handle<edm::DetSetVector<SiStripDigi> > digis;
   iEvent.getByLabel(_digicollection,digis);

   // loop on detector with digis

   int nshots=0;

   APVShotFinder apvsf(*digis,_zs);
   const std::vector<APVShot>& shots = apvsf.getShots();

   for(std::vector<APVShot>::const_iterator shot=shots.begin();shot!=shots.end();++shot) {
     if(shot->isGenuine()) {
       
	 ++nshots;
       
	 _whichAPV->Fill(shot->apvNumber());
	 _median->Fill(shot->median());
	 _stripMult->Fill(shot->nStrips());
	 _subDetector->Fill(shot->subDet());
	 uint32_t det=shot->detId();
	 tkhisto2->fill(det,shot->nStrips());;
	 tkhisto->add(det,1);

     }
   }

   _nShots->Fill(nshots);
   _nShotsVsTime->Fill(iEvent.orbitNumber(),nshots);
   

}

void 
APVShotsAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{}

void 
APVShotsAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
APVShotsAnalyzer::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
APVShotsAnalyzer::endJob() {

  edm::LogInfo("EndOfJob") << _nevents << " analyzed events";

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
  TrackerMap tkmap,tkmap2;

  tkmap.setPalette(1);
  tkmap2.setPalette(1);
  tkhisto->dumpInTkMap(&tkmap);
  tkhisto2->dumpInTkMap(&tkmap2);
  std::string tkshotmultmapname = "ShotMultiplicity_" + _suffix + ".png";
  tkmap.save(true,0,0,tkshotmultmapname);
  std::string tkstripmultmapname = "StripMultiplicity_" + _suffix + ".png";
  tkmap2.save(true,0,0,tkstripmultmapname);

  std::string rootmapname = "TKMap_"+_suffix+".root";
  tkhisto->save(rootmapname);
  tkhisto2->save(rootmapname);
}


//define this as a plug-in
DEFINE_FWK_MODULE(APVShotsAnalyzer);
