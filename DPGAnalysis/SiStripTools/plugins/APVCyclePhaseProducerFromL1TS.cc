// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      APVCyclePhaseProducerFromL1TS
//
/**\class APVCyclePhaseProducerFromL1TS APVCyclePhaseProducerFromL1TS.cc DPGAnalysis/SiStripTools/plugins/APVCyclePhaseProducerFromL1TS.cc

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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

//
// class decleration
//

class APVCyclePhaseProducerFromL1TS : public edm::EDProducer {
   public:
      explicit APVCyclePhaseProducerFromL1TS(const edm::ParameterSet&);
      ~APVCyclePhaseProducerFromL1TS();

private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  bool isBadRun(const unsigned int) const;
  void printConfiguration(std::stringstream& ss) const;
  
      // ----------member data ---------------------------

  edm::ESWatcher<SiStripConfObjectRcd> m_eswatcher;
  edm::EDGetTokenT<Level1TriggerScalersCollection> _l1tscollectionToken;
  const bool m_ignoreDB;
  const std::string m_rcdLabel;
  std::vector<std::string> _defpartnames;
  std::vector<int> _defphases;
  const bool _wantHistos;
  bool _useEC0;
  int _magicOffset;
  bool m_badRun;
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

  std::vector<std::pair<unsigned int, unsigned int> > m_badruns;

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
APVCyclePhaseProducerFromL1TS::APVCyclePhaseProducerFromL1TS(const edm::ParameterSet& iConfig):
  m_eswatcher(),
  _l1tscollectionToken(consumes<Level1TriggerScalersCollection>(iConfig.getParameter<edm::InputTag>("l1TSCollection"))),
  m_ignoreDB(iConfig.getUntrackedParameter<bool>("ignoreDB",false)),
  m_rcdLabel(iConfig.getUntrackedParameter<std::string>("recordLabel","apvphaseoffsets")),
  _defpartnames(iConfig.getParameter<std::vector<std::string> >("defaultPartitionNames")),
  _defphases(iConfig.getParameter<std::vector<int> >("defaultPhases")),
  _wantHistos(iConfig.getUntrackedParameter<bool>("wantHistos",false)),
  _useEC0(iConfig.getUntrackedParameter<bool>("useEC0",false)),
  _magicOffset(iConfig.getUntrackedParameter<int>("magicOffset",8)),
  m_badRun(false),
  m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin",250)),
  m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction",16)),
  m_rhm(consumesCollector()),
  _hsize(0),_hlresync(0),_hlOC0(0),_hlTE(0),_hlstart(0),_hlEC0(0),_hlHR(0),_hdlec0lresync(0),_hdlresynclHR(0),
  m_badruns(),
  _lastResync(-1),_lastHardReset(-1),_lastStart(-1),
  _lastEventCounter0(-1),_lastOrbitCounter0(-1),_lastTestEnable(-1)
{

  std::stringstream ss;
  printConfiguration(ss);
  edm::LogInfo("ConfigurationAtConstruction") << ss.str();
  

  produces<APVCyclePhaseCollection,edm::InEvent>();

  m_badruns.push_back(std::pair<unsigned int, unsigned int>(0,131767));
  m_badruns.push_back(std::pair<unsigned int, unsigned int>(193150,193733));

   //now do what ever other initialization is needed

  if(_wantHistos) {
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


}


APVCyclePhaseProducerFromL1TS::~APVCyclePhaseProducerFromL1TS()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
APVCyclePhaseProducerFromL1TS::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)

{

  // update the parameters from DB

  if(!m_ignoreDB && m_eswatcher.check(iSetup)) {
    edm::ESHandle<SiStripConfObject> confObj;
    iSetup.get<SiStripConfObjectRcd>().get(m_rcdLabel,confObj);

    std::stringstream summary;
    confObj->printDebug(summary);
    LogDebug("SiStripConfObjectSummary") << summary.str();

    _defpartnames = confObj->get<std::vector<std::string> >("defaultPartitionNames");
    _defphases    = confObj->get<std::vector<int> >("defaultPhases");
    _useEC0       = confObj->get<bool>("useEC0");
    m_badRun      = confObj->get<bool>("badRun");
    _magicOffset  = confObj->get<int>("magicOffset");

    std::stringstream ss;
    printConfiguration(ss);
    edm::LogInfo("UpdatedConfiguration") << ss.str();


  }

  if(_wantHistos) {

    m_rhm.beginRun(iRun);

    if(_hlresync && *_hlresync) {
      (*_hlresync)->GetXaxis()->SetTitle("Orbit");     (*_hlresync)->GetYaxis()->SetTitle("Events");
      (*_hlresync)->SetBit(TH1::kCanRebin);
    }

    if(_hlOC0 && *_hlOC0) {
      (*_hlOC0)->GetXaxis()->SetTitle("Orbit");     (*_hlOC0)->GetYaxis()->SetTitle("Events");
      (*_hlOC0)->SetBit(TH1::kCanRebin);
    }

    if(_hlTE && *_hlTE) {
      (*_hlTE)->GetXaxis()->SetTitle("Orbit");     (*_hlTE)->GetYaxis()->SetTitle("Events");
      (*_hlTE)->SetBit(TH1::kCanRebin);
    }

    if(_hlstart && *_hlstart) {
      (*_hlstart)->GetXaxis()->SetTitle("Orbit");     (*_hlstart)->GetYaxis()->SetTitle("Events");
      (*_hlstart)->SetBit(TH1::kCanRebin);
    }

    if(_hlEC0 && *_hlEC0) {
      (*_hlEC0)->GetXaxis()->SetTitle("Orbit");     (*_hlEC0)->GetYaxis()->SetTitle("Events");
      (*_hlEC0)->SetBit(TH1::kCanRebin);
    }

    if(_hlHR && *_hlHR) {
      (*_hlHR)->GetXaxis()->SetTitle("Orbit");     (*_hlHR)->GetYaxis()->SetTitle("Events");
      (*_hlHR)->SetBit(TH1::kCanRebin);
    }

    if(_hdlec0lresync && *_hdlec0lresync) {
      (*_hdlec0lresync)->GetXaxis()->SetTitle("lastEC0-lastResync");
    }

    if(_hdlresynclHR && *_hdlresynclHR) {
      (*_hdlresynclHR)->GetXaxis()->SetTitle("lastEC0-lastResync");
    }

  }

  if(isBadRun(iRun.run())) {
    LogDebug("UnreliableMissingL1TriggerScalers") <<
      "In this run L1TriggerScalers is missing or unreliable for phase determination: invlid phase will be returned";
  }

}


void
APVCyclePhaseProducerFromL1TS::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  std::auto_ptr<APVCyclePhaseCollection> apvphases(new APVCyclePhaseCollection() );


  std::vector<int> phases(_defphases.size(),APVCyclePhaseCollection::invalid);

  const std::vector<std::string>& partnames = _defpartnames;

  int phasechange = 0;


  Handle<Level1TriggerScalersCollection> l1ts;
  iEvent.getByToken(_l1tscollectionToken,l1ts);

  if(_wantHistos && _hsize && *_hsize) (*_hsize)->Fill(l1ts->size());

  // offset computation

  long long orbitoffset = 0;

  if(l1ts->size()>0) {

    if((*l1ts)[0].lastResync()!=0) {
      orbitoffset = _useEC0 ? (*l1ts)[0].lastEventCounter0() + _magicOffset : (*l1ts)[0].lastResync() + _magicOffset;
    }

    if(_wantHistos) {
      if(_hlresync && *_hlresync) (*_hlresync)->Fill((*l1ts)[0].lastResync());
      if(_hlOC0 && *_hlOC0) (*_hlOC0)->Fill((*l1ts)[0].lastOrbitCounter0());
      if(_hlTE && *_hlTE) (*_hlTE)->Fill((*l1ts)[0].lastTestEnable());
      if(_hlstart && *_hlstart) (*_hlstart)->Fill((*l1ts)[0].lastStart());
      if(_hlEC0 && *_hlEC0) (*_hlEC0)->Fill((*l1ts)[0].lastEventCounter0());
      if(_hlHR && *_hlHR) (*_hlHR)->Fill((*l1ts)[0].lastHardReset());
    }

    if(_lastResync != (*l1ts)[0].lastResync()) {
      _lastResync = (*l1ts)[0].lastResync();
      if(_wantHistos && _hdlec0lresync && *_hdlec0lresync) (*_hdlec0lresync)->Fill((*l1ts)[0].lastEventCounter0()-(*l1ts)[0].lastResync());
      LogDebug("TTCSignalReceived") << "New Resync at orbit " << _lastResync ;
    }
    if(_lastHardReset != (*l1ts)[0].lastHardReset()) {
      _lastHardReset = (*l1ts)[0].lastHardReset();
      if(_wantHistos && _hdlresynclHR && *_hdlresynclHR) (*_hdlresynclHR)->Fill((*l1ts)[0].lastResync()-(*l1ts)[0].lastHardReset());
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

    if(!isBadRun(iEvent.run())) {
      phasechange = ((long long)(orbitoffset*3564))%70;

      for(unsigned int ipart=0;ipart<phases.size();++ipart) {
	phases[ipart] = (_defphases[ipart]+phasechange)%70;
      }

    }
  }


  if(phases.size() < partnames.size() ) {
    // throw exception
    throw cms::Exception("InvalidAPVCyclePhases") << " Inconsistent phases/partitions vector sizes: "
					     << phases.size() << " "
					     << partnames.size();
  }

  for(unsigned int ipart=0;ipart<partnames.size();++ipart) {
    //    if(phases[ipart]>=0) {
      //      apvphases->get()[partnames[ipart]] = (phases[ipart]+phasechange)%70;
    apvphases->get()[partnames[ipart]] = phases[ipart];

      //    }
  }


  iEvent.put(apvphases);

}

bool
APVCyclePhaseProducerFromL1TS::isBadRun(const unsigned int run) const {

  for(std::vector<std::pair<unsigned int, unsigned int> >::const_iterator runpair = m_badruns.begin();runpair!=m_badruns.end();++runpair) {
    if( run >= runpair->first && run <= runpair->second) return true;
  }

  return m_badRun;

}

void
APVCyclePhaseProducerFromL1TS::printConfiguration(std::stringstream& ss) const {

  ss << _defpartnames.size() << " default partition names: ";
  for(std::vector<std::string>::const_iterator part=_defpartnames.begin();part!=_defpartnames.end();++part) {
    ss << *part << " ";
  }
  ss << std::endl;
  ss << _defphases.size() << " default phases: ";
  for(std::vector<int>::const_iterator phase=_defphases.begin();phase!=_defphases.end();++phase) {
    ss << *phase << " ";
  }
  ss << std::endl;
  ss << " Magic offset: " << _magicOffset << std::endl;
  ss << " use ECO: " << _useEC0 << std::endl;
  ss << " bad run: " << m_badRun << std::endl;

}
//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseProducerFromL1TS);
