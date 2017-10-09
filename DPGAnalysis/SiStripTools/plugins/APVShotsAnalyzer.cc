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

#include "DQM/SiStripCommon/interface/APVShotFinder.h"
#include "DQM/SiStripCommon/interface/APVShot.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
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
  virtual void beginJob() override ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  void updateDetCabling( const edm::EventSetup& setup );

      // ----------member data ---------------------------

  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > _digicollectionToken;
  edm::EDGetTokenT<EventWithHistory> _historyProductToken;
  edm::EDGetTokenT<APVCyclePhaseCollection> _apvphasecollToken;
  const std::string _phasepart;
  bool _zs;
  std::string _suffix;
  int _nevents;

  TH1F* _nShots;
  TH1F* _whichAPV;
  TH1F* _stripMult;
  TH1F* _median;
  TH1F* _subDetector;
  TH1F* _fed;
  TH2F* _channelvsfed;

  TProfile* _nShotsbxcycle;
  TProfile* _nShotsdbx;
  TProfile* _nShotsdbxincycle;
  TProfile* _nShotsbxcycleprev;
  TProfile* _nShotsdbxprev;
  TProfile* _nShotsdbxincycleprev;

  TH2F* _medianVsFED;
  TH2F* _nShotsVsFED;

  RunHistogramManager _rhm;

  TH1F** _nShotsrun;
  TProfile** _nShotsVsTimerun;
  TH1F** _whichAPVrun;
  TH1F** _stripMultrun;
  TH1F** _medianrun;
  TH1F** _subDetectorrun;
  TH1F** _fedrun;

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
  _digicollectionToken(consumes<edm::DetSetVector<SiStripDigi> >(iConfig.getParameter<edm::InputTag>("digiCollection"))),
  _historyProductToken(consumes<EventWithHistory>(iConfig.getParameter<edm::InputTag>("historyProduct"))),
  _apvphasecollToken(consumes<APVCyclePhaseCollection>(iConfig.getParameter<edm::InputTag>("apvPhaseCollection"))),
  _phasepart(iConfig.getUntrackedParameter<std::string>("phasePartition","None")),
  _zs(iConfig.getUntrackedParameter<bool>("zeroSuppressed",true)),
  _suffix(iConfig.getParameter<std::string>("mapSuffix")),
  _nevents(0),
  _rhm(consumesCollector()),
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

 _whichAPV = tfserv->make<TH1F>("whichAPV","APV with shots",6,-0.5,5.5);
 _whichAPV->GetXaxis()->SetTitle("APV");  _whichAPV->GetYaxis()->SetTitle("Shots");

 _stripMult = tfserv->make<TH1F>("stripMultiplicity","Shot Strip Multiplicity",129,-0.5,128.5);
 _stripMult->GetXaxis()->SetTitle("Number of Strips");  _stripMult->GetYaxis()->SetTitle("Shots");

 _median = tfserv->make<TH1F>("median","APV Shot charge median",256,-0.5,255.5);
 _median->GetXaxis()->SetTitle("Charge [ADC]");  _median->GetYaxis()->SetTitle("Shots");

 _subDetector = tfserv->make<TH1F>("subDets","SubDetector Shot distribution",10,-0.5,9.5);
 _subDetector->GetYaxis()->SetTitle("Shots");

 _nShotsbxcycle = tfserv->make<TProfile>("nShotsBXcycle","Number of shots vs APV cycle bin",70,-0.5,69.5);
 _nShotsbxcycle->GetXaxis()->SetTitle("Event BX mod(70)");  _nShotsbxcycle->GetYaxis()->SetTitle("APV shots");

 _nShotsdbx = tfserv->make<TProfile>("nShotsDBX","Number of shots vs #Delta(BX)",1000,-0.5,999.5);
 _nShotsdbx->GetXaxis()->SetTitle("Event #Delta(BX)");  _nShotsdbx->GetYaxis()->SetTitle("APV shots");

 _nShotsdbxincycle = tfserv->make<TProfile>("nShotsDBXincycle","Number of shots vs #Delta(BX) w.r.t. APV cycle",1000,-0.5,999.5);
 _nShotsdbxincycle->GetXaxis()->SetTitle("Event #Delta(BX) w.r.t. APV cycle");  _nShotsdbxincycle->GetYaxis()->SetTitle("APV shots");

 _nShotsbxcycleprev = tfserv->make<TProfile>("nShotsBXcycleprev","Number of shots vs APV cycle bin of previous L1A",70,-0.5,69.5);
 _nShotsbxcycleprev->GetXaxis()->SetTitle("Previous L1A BX mod(70)");  _nShotsbxcycleprev->GetYaxis()->SetTitle("APV shots");

 _nShotsdbxprev = tfserv->make<TProfile>("nShotsDBXprev","Number of shots vs #Delta(BX) of previous L1A",1000,-0.5,999.5);
 _nShotsdbxprev->GetXaxis()->SetTitle("Previous L1A #Delta(BX)");  _nShotsdbxprev->GetYaxis()->SetTitle("APV shots");

 _nShotsdbxincycleprev = tfserv->make<TProfile>("nShotsDBXincycleprev","Number of shots vs #Delta(BX) w.r.t. APV cycle of previous L1A",1000,-0.5,999.5);
 _nShotsdbxincycleprev->GetXaxis()->SetTitle("Previous L1A #Delta(BX) w.r.t. APV cycle");  _nShotsdbxincycleprev->GetYaxis()->SetTitle("APV shots");

 _nShotsrun = _rhm.makeTH1F("nShotsrun","Number of Shots per event",200,-0.5,199.5);
 _nShotsVsTimerun  = _rhm.makeTProfile("nShotsVsTimerun","Mean number of shots vs orbit number",4*500,0,500*262144);
 _whichAPVrun = _rhm.makeTH1F("whichAPVrun","APV with shots",6,-0.5,5.5);
 _stripMultrun = _rhm.makeTH1F("stripMultiplicityrun","Shot Strip Multiplicity",129,-0.5,128.5);
 _medianrun = _rhm.makeTH1F("medianrun","APV Shot charge median",256,-0.5,255.5);
 _subDetectorrun = _rhm.makeTH1F("subDetsrun","SubDetector Shot distribution",10,-0.5,9.5);

 if (_useCabling) {
   _fed = tfserv->make<TH1F>("fed","FED Shot distribution",440,50,490);
   _fed->GetYaxis()->SetTitle("Shots");
   _fedrun = _rhm.makeTH1F("fedrun","FED Shot distribution",440,50,490);

   _channelvsfed = tfserv->make<TH2F>("channelvsfed","Channel vs FED Shot distribution",440,50,490,97,-0.5,96.5);
   _channelvsfed->GetXaxis()->SetTitle("FED");    _channelvsfed->GetYaxis()->SetTitle("Channel");


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

   edm::Handle<EventWithHistory> he;
   iEvent.getByToken(_historyProductToken,he);

   edm::Handle<APVCyclePhaseCollection> apvphase;
   iEvent.getByToken(_apvphasecollToken,apvphase);

   int thephase = APVCyclePhaseCollection::invalid;
   if(apvphase.isValid() && !apvphase.failedToGet()) {
     thephase = apvphase->getPhase(_phasepart);
   }
   bool isphaseok = (thephase!=APVCyclePhaseCollection::invalid &&
		     thephase!=APVCyclePhaseCollection::multiphase &&
		     thephase!=APVCyclePhaseCollection::nopartition);

   Handle<edm::DetSetVector<SiStripDigi> > digis;
   iEvent.getByToken(_digicollectionToken,digis);

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

	 int apvPair = shot->apvNumber()/2;
	 LogDebug("APVPair") << apvPair;

	 const FedChannelConnection& theConn = _detCabling->getConnection( det , apvPair);

	 int lChannelId = -1;
	 int thelFEDId = -1;
	 if(theConn.isConnected()) {
	   lChannelId = theConn.fedCh();
	   thelFEDId = theConn.fedId();
	 }
	 else {
	   edm::LogWarning("ConnectionNotFound") << "connection of det " << det << " APV pair " << apvPair << " not found";
	 }
	 LogDebug("FED channels") << thelFEDId << " " << lChannelId ;

         const std::vector<const FedChannelConnection *> & conns = _detCabling->getConnections( det );

	 if (!(conns.size())) continue;
	 uint16_t lFedId = 0;
	 for (uint32_t ch = 0; ch<conns.size(); ch++) {
	   if(conns[ch] && conns[ch]->isConnected()) {
	     LogDebug("Dump") << *(conns[ch]);
	     LogDebug("ReadyForFEDid") << "Ready for FED id " << ch;
	     lFedId = conns[ch]->fedId();
	     LogDebug("FEDid") << "obtained FED id " << ch << " " << lFedId;
	     //uint16_t lFedCh = conns[ch]->fedCh();

	     if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX){
	       edm::LogWarning("InvalidFEDid") << lFedId << " for detid " << det << " connection " << ch;
	       continue;
	     }
	     else break;
	   }
	 }
	 if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX){
	   edm::LogWarning("NoValidFEDid") << lFedId <<  "found for detid " << det;
	   continue;
	 }

	 if(lFedId != thelFEDId) {
	   edm::LogWarning("FEDidMismatch") << " Mismatch in FED id for det " << det << " APV pair "
					    << apvPair << " : " << lFedId << " vs " << thelFEDId;
	 }

	 LogDebug("FillingArray") << nshotsperFed.size() << " " << lFedId-sistrip::FED_ID_MIN;
	 ++nshotsperFed[lFedId-sistrip::FED_ID_MIN];

	 LogDebug("ReadyToBeFilled") << " ready to be filled with " << thelFEDId << " " << lChannelId;
	 _channelvsfed->Fill(thelFEDId,lChannelId);
	 LogDebug("Filled") << " filled with " << thelFEDId << " " << lChannelId;

	 _fed->Fill(lFedId);

	 if(_fedrun && *_fedrun) (*_fedrun)->Fill(lFedId);
	 _medianVsFED->Fill(lFedId,shot->median());


       }

       ++nshots;


       _whichAPV->Fill(shot->apvNumber());
       _median->Fill(shot->median());
       _stripMult->Fill(shot->nStrips());
       _subDetector->Fill(shot->subDet());

       if(_whichAPVrun && *_whichAPVrun) (*_whichAPVrun)->Fill(shot->apvNumber());
       if(_medianrun && *_medianrun) (*_medianrun)->Fill(shot->median());
       if(_stripMultrun && *_stripMultrun) (*_stripMultrun)->Fill(shot->nStrips());
       if(_subDetectorrun && *_subDetectorrun) (*_subDetectorrun)->Fill(shot->subDet());

       tkhisto2->fill(det,shot->nStrips());;
       tkhisto->add(det,1);



     }
   }

     _nShots->Fill(nshots);
     if(_nShotsrun && *_nShotsrun) (*_nShotsrun)->Fill(nshots);

   _nShotsdbx->Fill(he->deltaBX(),nshots);
   _nShotsdbxprev->Fill(he->deltaBX(),nshots);
   if(isphaseok) {
     _nShotsbxcycle->Fill(he->absoluteBXinCycle(thephase)%70,nshots);
     _nShotsdbxincycle->Fill(he->deltaBXinCycle(thephase),nshots);
     _nShotsbxcycleprev->Fill(he->absoluteBXinCycle(1,thephase)%70,nshots);
     _nShotsdbxincycleprev->Fill(he->deltaBXinCycle(1,2,thephase),nshots);
   }

   if (_useCabling){
     for (uint16_t lFed(0); lFed<lNumFeds; lFed++){
       _nShotsVsFED->Fill(lFed+sistrip::FED_ID_MIN,nshotsperFed[lFed]);
     }
   }

   if(_nShotsVsTimerun && *_nShotsVsTimerun) (*_nShotsVsTimerun)->Fill(iEvent.orbitNumber(),nshots);


}

void
APVShotsAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{


  _rhm.beginRun(iRun);

  if(_nShotsrun && *_nShotsrun) {
    (*_nShotsrun)->GetXaxis()->SetTitle("Shots");  (*_nShotsrun)->GetYaxis()->SetTitle("Events");
    (*_nShotsrun)->StatOverflows(kTRUE);
  }

  if(_nShotsVsTimerun && *_nShotsVsTimerun) {
    (*_nShotsVsTimerun)->GetXaxis()->SetTitle("Orbit");  (*_nShotsVsTimerun)->GetYaxis()->SetTitle("Number of Shots");
    (*_nShotsVsTimerun)->SetCanExtend(TH1::kXaxis);
  }

  if(_whichAPVrun && *_whichAPVrun) {
    (*_whichAPVrun)->GetXaxis()->SetTitle("APV");  (*_whichAPVrun)->GetYaxis()->SetTitle("Shots");
  }

  if(_stripMultrun && *_stripMultrun) {
    (*_stripMultrun)->GetXaxis()->SetTitle("Number of Strips");  (*_stripMultrun)->GetYaxis()->SetTitle("Shots");
  }

  if(_medianrun && *_medianrun) {
    (*_medianrun)->GetXaxis()->SetTitle("Charge [ADC]");  (*_medianrun)->GetYaxis()->SetTitle("Shots");
  }

  if(_subDetectorrun && *_subDetectorrun) {
    (*_subDetectorrun)->GetYaxis()->SetTitle("Shots");
  }

  if (_useCabling) {
    if(_fedrun && *_fedrun) {
      (*_fedrun)->GetYaxis()->SetTitle("Shots");
    }
  }

}

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
