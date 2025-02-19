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
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DPGAnalysis/SiStripTools/interface/APVShotFinder.h"
#include "DPGAnalysis/SiStripTools/interface/APVShot.h"

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
//***************************************************

//******** includes for the cabling *************
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
//***************************************************


//
// class decleration
//

class APVShotsAnalyzer : public edm::EDAnalyzer {
public:
  explicit APVShotsAnalyzer(const edm::ParameterSet&);
  ~APVShotsAnalyzer();


private:
  virtual void beginJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void updateDetCabling( const edm::EventSetup& setup );

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
  TH1F* _fed;

  TH2F* _medianVsFED;
  TH2F* _nShotsVsFED;


  TkHistoMap *tkhisto,*tkhisto2; 

  // DetCabling
  bool _useCabling;
  uint32_t _cacheIdDet;  //!< DB cache ID used to establish if the cabling has changed during the run.
  const SiStripDetCabling* _detCabling;  //!< The cabling object.
  
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
  _nevents(0),
  _useCabling(iConfig.getUntrackedParameter<bool>("useCabling",true)),
  _cacheIdDet(0),
  _detCabling(0)
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

 if (_useCabling) {
   _fed = tfserv->make<TH1F>("fed","FED Shot distribution",440,50,490);
   _fed->GetYaxis()->SetTitle("Shots");

   _nShotsVsFED = tfserv->make<TH2F>("nShotsVsFED","Number of Shots per event vs fedid",440,50,490,200,-0.5,199.5);
   _nShotsVsFED->GetXaxis()->SetTitle("fedId");  _nShots->GetYaxis()->SetTitle("Shots");  _nShots->GetZaxis()->SetTitle("Events");
   _nShotsVsFED->StatOverflows(kTRUE);

   _medianVsFED = tfserv->make<TH2F>("medianVsFED","APV Shot charge median vs fedid",440,50,490,256,-0.5,255.5);
   _medianVsFED->GetXaxis()->SetTitle("fedId");_medianVsFED->GetYaxis()->SetTitle("Charge [ADC]");  _median->GetZaxis()->SetTitle("Shots");
 }

 tkhisto      =new TkHistoMap("ShotMultiplicity","ShotMultiplicity",-1); 
 tkhisto2      =new TkHistoMap("StripMultiplicity","StripMultiplicity",-1); 
}


APVShotsAnalyzer::~APVShotsAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if ( _detCabling ) _detCabling = 0;

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
APVShotsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if (_useCabling){
     //retrieve cabling
     updateDetCabling( iSetup );
   }

   _nevents++;

   Handle<edm::DetSetVector<SiStripDigi> > digis;
   iEvent.getByLabel(_digicollection,digis);

   // loop on detector with digis

   int nshots=0;
   std::vector<int> nshotsperFed;

   const uint16_t lNumFeds = sistrip::FED_ID_MAX-sistrip::FED_ID_MIN+1;
   if (_useCabling){
     nshotsperFed.resize(lNumFeds,0);
   }

   APVShotFinder apvsf(*digis,_zs);
   const std::vector<APVShot>& shots = apvsf.getShots();

   for(std::vector<APVShot>::const_iterator shot=shots.begin();shot!=shots.end();++shot) {
     if(shot->isGenuine()) {

       //get the fedid from the detid

       uint32_t det=shot->detId();
       if (_useCabling){
         const std::vector<const FedChannelConnection *> & conns = _detCabling->getConnections( det );

	 if (!(conns.size())) continue;
	 uint16_t lFedId = 0;
	 for (uint32_t ch = 0; ch<conns.size(); ch++) {
           lFedId = conns[ch]->fedId();
           //uint16_t lFedCh = conns[ch]->fedCh();
   
	   if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX){
	     std::cout << " -- Invalid fedid " << lFedId << " for detid " << det << " connection " << ch << std::endl;
	     continue;
	   }
	   else break;
	 }

	 if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX){
	   std::cout << " -- No valid fedid (=" << lFedId << ") found for detid " << det << std::endl;
	   continue;
	 }
	 ++nshotsperFed[lFedId-sistrip::FED_ID_MIN];
	 _fed->Fill(lFedId);
	 _medianVsFED->Fill(lFedId,shot->median());


       }

       ++nshots;


       _whichAPV->Fill(shot->apvNumber());
       _median->Fill(shot->median());
       _stripMult->Fill(shot->nStrips());
       _subDetector->Fill(shot->subDet());
       tkhisto2->fill(det,shot->nStrips());;
       tkhisto->add(det,1);



     }
   }

   _nShots->Fill(nshots);
   if (_useCabling){
     for (uint16_t lFed(0); lFed<lNumFeds; lFed++){
       _nShotsVsFED->Fill(lFed+sistrip::FED_ID_MIN,nshotsperFed[lFed]);
     }
   }

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
APVShotsAnalyzer::beginJob()
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


void APVShotsAnalyzer::updateDetCabling( const edm::EventSetup& setup )
{
  if (_useCabling){
    uint32_t cache_id = setup.get<SiStripDetCablingRcd>().cacheIdentifier();//.get( cabling_ );
   
    if ( _cacheIdDet != cache_id ) { // If the cache ID has changed since the last update...
      // Update the cabling object
      edm::ESHandle<SiStripDetCabling> c;
      setup.get<SiStripDetCablingRcd>().get( c );
      _detCabling = c.product();
      _cacheIdDet = cache_id;
    } // end of new cache ID check
  }
}




//define this as a plug-in
DEFINE_FWK_MODULE(APVShotsAnalyzer);
