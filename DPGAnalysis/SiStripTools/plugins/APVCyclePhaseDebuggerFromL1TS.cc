// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      APVCyclePhaseDebuggerFromL1TS
//
/**\class APVCyclePhaseDebuggerFromL1TS APVCyclePhaseDebuggerFromL1TS.cc DPGAnalysis/SiStripTools/plugins/APVCyclePhaseDebuggerFromL1TS.cc

 Description: EDproducer for APVCyclePhaseCollection which uses the configuration file to assign a phase to the run

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Jan 12 09:05:45 CET 2009
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

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

//
// class decleration
//

class APVCyclePhaseDebuggerFromL1TS : public edm::EDAnalyzer {
public:
  explicit APVCyclePhaseDebuggerFromL1TS(const edm::ParameterSet&);
  ~APVCyclePhaseDebuggerFromL1TS();
  
  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );

private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<Level1TriggerScalersCollection> _l1tscollectionToken;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;

  RunHistogramManager m_rhm;

  TH1F** _hsize;
  TH1F** _hlresync;
  TH1F** _hlOC0;
  TH1F** _hlTE;
  TH1F** _hlstart;
  TH1F** _hlEC0;
  TH1F** _hlHR;

  TH1F** _hdlec0lresync;
  TH1F** _hdlresynclHR;

  long long _lastResync;
  long long _lastHardReset;
  long long _lastStart;
  long long _lastEventCounter0;
  long long _lastOrbitCounter0;
  long long _lastTestEnable;


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
APVCyclePhaseDebuggerFromL1TS::APVCyclePhaseDebuggerFromL1TS(const edm::ParameterSet& iConfig):
  _l1tscollectionToken(consumes<Level1TriggerScalersCollection>(iConfig.getParameter<edm::InputTag>("l1TSCollection"))),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",250)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",16)),
  m_rhm(consumesCollector()),
  _hsize(0),_hlresync(0),_hlOC0(0),_hlTE(0),_hlstart(0),_hlEC0(0),_hlHR(0),_hdlec0lresync(0),_hdlresynclHR(0),
  _lastResync(-1),_lastHardReset(-1),_lastStart(-1),
  _lastEventCounter0(-1),_lastOrbitCounter0(-1),_lastTestEnable(-1)
{

   //now do what ever other initialization is needed

  _hsize = m_rhm.makeTH1F("size","Level1TriggerScalers Collection size",20,-0.5,19.5);
  
  _hlresync = m_rhm.makeTH1F("lresync","Orbit of last resync",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hlOC0 = m_rhm.makeTH1F("lOC0","Orbit of last OC0",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hlTE = m_rhm.makeTH1F("lTE","Orbit of last TestEnable",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hlstart = m_rhm.makeTH1F("lstart","Orbit of last Start",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hlEC0 = m_rhm.makeTH1F("lEC0","Orbit of last EC0",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hlHR = m_rhm.makeTH1F("lHR","Orbit of last HardReset",m_LSfrac*m_maxLS,0,m_maxLS*262144);
  _hdlec0lresync = m_rhm.makeTH1F("dlec0lresync","Orbit difference EC0-Resync",4000,-1999.5,2000.5);
  _hdlresynclHR = m_rhm.makeTH1F("dlresynclHR","Orbit difference Resync-HR",4000,-1999.5,2000.5);
  
}


APVCyclePhaseDebuggerFromL1TS::~APVCyclePhaseDebuggerFromL1TS()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
APVCyclePhaseDebuggerFromL1TS::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)

{

  // update the parameters from DB


  m_rhm.beginRun(iRun);
  
  if(_hlresync && *_hlresync) {
    (*_hlresync)->GetXaxis()->SetTitle("Orbit");     (*_hlresync)->GetYaxis()->SetTitle("Events");
    (*_hlresync)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hlOC0 && *_hlOC0) {
    (*_hlOC0)->GetXaxis()->SetTitle("Orbit");     (*_hlOC0)->GetYaxis()->SetTitle("Events");
    (*_hlOC0)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hlTE && *_hlTE) {
    (*_hlTE)->GetXaxis()->SetTitle("Orbit");     (*_hlTE)->GetYaxis()->SetTitle("Events");
    (*_hlTE)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hlstart && *_hlstart) {
    (*_hlstart)->GetXaxis()->SetTitle("Orbit");     (*_hlstart)->GetYaxis()->SetTitle("Events");
    (*_hlstart)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hlEC0 && *_hlEC0) {
    (*_hlEC0)->GetXaxis()->SetTitle("Orbit");     (*_hlEC0)->GetYaxis()->SetTitle("Events");
    (*_hlEC0)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hlHR && *_hlHR) {
    (*_hlHR)->GetXaxis()->SetTitle("Orbit");     (*_hlHR)->GetYaxis()->SetTitle("Events");
    (*_hlHR)->SetCanExtend(TH1::kXaxis);
  }
  
  if(_hdlec0lresync && *_hdlec0lresync) {
    (*_hdlec0lresync)->GetXaxis()->SetTitle("lastEC0-lastResync");
  }
  
  if(_hdlresynclHR && *_hdlresynclHR) {
    (*_hdlresynclHR)->GetXaxis()->SetTitle("lastEC0-lastResync");
  }
  
}


void
APVCyclePhaseDebuggerFromL1TS::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;


  Handle<Level1TriggerScalersCollection> l1ts;
  iEvent.getByToken(_l1tscollectionToken,l1ts);

  if(_hsize && *_hsize) (*_hsize)->Fill(l1ts->size());

  // offset computation

  if(l1ts->size()>0) {

    if(_hlresync && *_hlresync) (*_hlresync)->Fill((*l1ts)[0].lastResync());
    if(_hlOC0 && *_hlOC0) (*_hlOC0)->Fill((*l1ts)[0].lastOrbitCounter0());
    if(_hlTE && *_hlTE) (*_hlTE)->Fill((*l1ts)[0].lastTestEnable());
    if(_hlstart && *_hlstart) (*_hlstart)->Fill((*l1ts)[0].lastStart());
    if(_hlEC0 && *_hlEC0) (*_hlEC0)->Fill((*l1ts)[0].lastEventCounter0());
    if(_hlHR && *_hlHR) (*_hlHR)->Fill((*l1ts)[0].lastHardReset());

    if(_lastResync != (*l1ts)[0].lastResync()) {
      _lastResync = (*l1ts)[0].lastResync();
      if(_hdlec0lresync && *_hdlec0lresync) (*_hdlec0lresync)->Fill((*l1ts)[0].lastEventCounter0()-(*l1ts)[0].lastResync());
      LogDebug("TTCSignalReceived") << "New Resync at orbit " << _lastResync ;
    }
    if(_lastHardReset != (*l1ts)[0].lastHardReset()) {
      _lastHardReset = (*l1ts)[0].lastHardReset();
      if(_hdlresynclHR && *_hdlresynclHR) (*_hdlresynclHR)->Fill((*l1ts)[0].lastResync()-(*l1ts)[0].lastHardReset());
      LogDebug("TTCSignalReceived") << "New HardReset at orbit " << _lastHardReset ;
    }
    if(_lastTestEnable != (*l1ts)[0].lastTestEnable()) {
      _lastTestEnable = (*l1ts)[0].lastTestEnable();
      //      LogDebug("TTCSignalReceived") << "New TestEnable at orbit " << _lastTestEnable ;
    }
    if(_lastOrbitCounter0 != (*l1ts)[0].lastOrbitCounter0()) {
      _lastOrbitCounter0 = (*l1ts)[0].lastOrbitCounter0();
      LogDebug("TTCSignalReceived") << "New OrbitCounter0 at orbit " << _lastOrbitCounter0 ;
    }
    if(_lastEventCounter0 != (*l1ts)[0].lastEventCounter0()) {
      _lastEventCounter0 = (*l1ts)[0].lastEventCounter0();
      LogDebug("TTCSignalReceived") << "New EventCounter0 at orbit " << _lastEventCounter0 ;
    }
    if(_lastStart != (*l1ts)[0].lastStart()) {
      _lastStart = (*l1ts)[0].lastStart();
      LogDebug("TTCSignalReceived") << "New Start at orbit " << _lastStart ;
    }

  }


}

void
APVCyclePhaseDebuggerFromL1TS::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1TSCollection",edm::InputTag("scalersRawToDigi"));
  descriptions.add("l1TSDebugger",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseDebuggerFromL1TS);
