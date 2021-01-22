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
#include "FWCore/Framework/interface/stream/EDProducer.h"

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

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

//
// class declaration
//

class APVCyclePhaseProducerFromL1TS : public edm::stream::EDProducer<> {
public:
  explicit APVCyclePhaseProducerFromL1TS(const edm::ParameterSet&);
  ~APVCyclePhaseProducerFromL1TS() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  bool isBadRun(const unsigned int) const;
  void printConfiguration(std::stringstream& ss) const;

  // ----------member data ---------------------------

  const bool m_ignoreDB;
  edm::ESWatcher<SiStripConfObjectRcd> m_eswatcher;
  edm::EDGetTokenT<Level1TriggerScalersCollection> _l1tscollectionToken;
  edm::EDGetTokenT<TCDSRecord> _tcdsRecordToken;
  edm::ESGetToken<SiStripConfObject, SiStripConfObjectRcd> _confObjectToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> _tTopoToken;
  std::vector<std::string> _defpartnames;
  std::vector<int> _defphases;
  bool _useEC0;
  int _magicOffset;
  bool m_badRun;
  const bool _forceSCAL;

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
APVCyclePhaseProducerFromL1TS::APVCyclePhaseProducerFromL1TS(const edm::ParameterSet& iConfig)
    : m_ignoreDB(iConfig.getUntrackedParameter<bool>("ignoreDB", false)),
      m_eswatcher(),
      _l1tscollectionToken(
          consumes<Level1TriggerScalersCollection>(iConfig.getParameter<edm::InputTag>("l1TSCollection"))),
      _tcdsRecordToken(consumes<TCDSRecord>(iConfig.getParameter<edm::InputTag>("tcdsRecordLabel"))),
      _confObjectToken((!m_ignoreDB)
                           ? decltype(_confObjectToken){esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{
                                 "", iConfig.getUntrackedParameter<std::string>("recordLabel", "apvphaseoffsets")})}
                           : decltype(_confObjectToken){}),
      _tTopoToken((!m_ignoreDB) ? decltype(_tTopoToken){esConsumes<edm::Transition::BeginRun>()}
                                : decltype(_tTopoToken){}),
      _defpartnames(iConfig.getParameter<std::vector<std::string> >("defaultPartitionNames")),
      _defphases(iConfig.getParameter<std::vector<int> >("defaultPhases")),
      _useEC0(iConfig.getUntrackedParameter<bool>("useEC0", false)),
      _magicOffset(iConfig.getUntrackedParameter<int>("magicOffset", 8)),
      m_badRun(false),
      _forceSCAL(iConfig.getParameter<bool>("forceSCAL")),
      m_badruns(),
      _lastResync(-1),
      _lastHardReset(-1),
      _lastStart(-1),
      _lastEventCounter0(-1),
      _lastOrbitCounter0(-1),
      _lastTestEnable(-1) {
  std::stringstream ss;
  printConfiguration(ss);
  edm::LogInfo("ConfigurationAtConstruction") << ss.str();

  produces<APVCyclePhaseCollection, edm::InEvent>();

  m_badruns.push_back(std::pair<unsigned int, unsigned int>(0, 131767));
  m_badruns.push_back(std::pair<unsigned int, unsigned int>(193150, 193733));

  //now do what ever other initialization is needed
}

APVCyclePhaseProducerFromL1TS::~APVCyclePhaseProducerFromL1TS() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void APVCyclePhaseProducerFromL1TS::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)

{
  // update the parameters from DB

  if (!m_ignoreDB && m_eswatcher.check(iSetup)) {
    const auto& confObj = iSetup.getData(_confObjectToken);

    std::stringstream summary;
    confObj.printDebug(summary, &iSetup.getData(_tTopoToken));
    LogDebug("SiStripConfObjectSummary") << summary.str();

    _defpartnames = confObj.get<std::vector<std::string> >("defaultPartitionNames");
    _defphases = confObj.get<std::vector<int> >("defaultPhases");
    _useEC0 = confObj.get<bool>("useEC0");
    m_badRun = confObj.get<bool>("badRun");
    _magicOffset = confObj.get<int>("magicOffset");

    std::stringstream ss;
    printConfiguration(ss);
    edm::LogInfo("UpdatedConfiguration") << ss.str();
  }

  if (isBadRun(iRun.run())) {
    LogDebug("UnreliableMissingL1TriggerScalers") << "In this run L1TriggerScalers is missing or unreliable for phase "
                                                     "determination: invlid phase will be returned";
  }
}

void APVCyclePhaseProducerFromL1TS::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<APVCyclePhaseCollection> apvphases(new APVCyclePhaseCollection());

  std::vector<int> phases(_defphases.size(), APVCyclePhaseCollection::invalid);

  const std::vector<std::string>& partnames = _defpartnames;

  int phasechange = 0;

  Handle<Level1TriggerScalersCollection> l1ts;
  iEvent.getByToken(_l1tscollectionToken, l1ts);
  Handle<TCDSRecord> tcds_pIn;
  iEvent.getByToken(_tcdsRecordToken, tcds_pIn);
  bool useTCDS(tcds_pIn.isValid() && !_forceSCAL);
  const auto* tcdsRecord = useTCDS ? tcds_pIn.product() : nullptr;
  // offset computation

  long long orbitoffset = 0;

  if (useTCDS && iEvent.eventAuxiliary().isRealData()) {
    // l1ts->empty() always retuns false (last commit as of today: https://github.com/cms-sw/cmssw/commit/f4694d795d4b268d541c633dfb68283d889264b0 ), so the check is likely not necessary---and TCDSRecord hasn't anything similar
    if (tcdsRecord->getLastResync() != 0) {
      orbitoffset =
          _useEC0 ? tcdsRecord->getLastEventCounter0() + _magicOffset : tcdsRecord->getLastResync() + _magicOffset;
    }

    if (_lastResync != tcdsRecord->getLastResync()) {
      _lastResync = tcdsRecord->getLastResync();
      LogDebug("TTCSignalReceived") << "New Resync at orbit " << _lastResync;
    }
    if (_lastHardReset != tcdsRecord->getLastHardReset()) {
      _lastHardReset = tcdsRecord->getLastHardReset();
      LogDebug("TTCSignalReceived") << "New HardReset at orbit " << _lastHardReset;
    }
    if (_lastTestEnable != tcdsRecord->getLastTestEnable()) {
      _lastTestEnable = tcdsRecord->getLastTestEnable();
      //      LogDebug("TTCSignalReceived") << "New TestEnable at orbit " << _lastTestEnable ;
    }
    if (_lastOrbitCounter0 != tcdsRecord->getLastOrbitCounter0()) {
      _lastOrbitCounter0 = tcdsRecord->getLastOrbitCounter0();
      LogDebug("TTCSignalReceived") << "New OrbitCounter0 at orbit " << _lastOrbitCounter0;
    }
    if (_lastEventCounter0 != tcdsRecord->getLastEventCounter0()) {
      _lastEventCounter0 = tcdsRecord->getLastEventCounter0();
      LogDebug("TTCSignalReceived") << "New EventCounter0 at orbit " << _lastEventCounter0;
    }
    if (_lastStart != tcdsRecord->getLastStart()) {
      _lastStart = tcdsRecord->getLastStart();
      LogDebug("TTCSignalReceived") << "New Start at orbit " << _lastStart;
    }

    if (!isBadRun(iEvent.run())) {
      phasechange = ((long long)(orbitoffset * 3564)) % 70;

      for (unsigned int ipart = 0; ipart < phases.size(); ++ipart) {
        phases[ipart] = (_defphases[ipart] + phasechange) % 70;
      }
    }

  } else {
    if (!l1ts->empty()) {
      if ((*l1ts)[0].lastResync() != 0) {
        orbitoffset = _useEC0 ? (*l1ts)[0].lastEventCounter0() + _magicOffset : (*l1ts)[0].lastResync() + _magicOffset;
      }

      if (_lastResync != (*l1ts)[0].lastResync()) {
        _lastResync = (*l1ts)[0].lastResync();
        LogDebug("TTCSignalReceived") << "New Resync at orbit " << _lastResync;
      }
      if (_lastHardReset != (*l1ts)[0].lastHardReset()) {
        _lastHardReset = (*l1ts)[0].lastHardReset();
        LogDebug("TTCSignalReceived") << "New HardReset at orbit " << _lastHardReset;
      }
      if (_lastTestEnable != (*l1ts)[0].lastTestEnable()) {
        _lastTestEnable = (*l1ts)[0].lastTestEnable();
        //      LogDebug("TTCSignalReceived") << "New TestEnable at orbit " << _lastTestEnable ;
      }
      if (_lastOrbitCounter0 != (*l1ts)[0].lastOrbitCounter0()) {
        _lastOrbitCounter0 = (*l1ts)[0].lastOrbitCounter0();
        LogDebug("TTCSignalReceived") << "New OrbitCounter0 at orbit " << _lastOrbitCounter0;
      }
      if (_lastEventCounter0 != (*l1ts)[0].lastEventCounter0()) {
        _lastEventCounter0 = (*l1ts)[0].lastEventCounter0();
        LogDebug("TTCSignalReceived") << "New EventCounter0 at orbit " << _lastEventCounter0;
      }
      if (_lastStart != (*l1ts)[0].lastStart()) {
        _lastStart = (*l1ts)[0].lastStart();
        LogDebug("TTCSignalReceived") << "New Start at orbit " << _lastStart;
      }

      if (!isBadRun(iEvent.run())) {
        phasechange = ((long long)(orbitoffset * 3564)) % 70;

        for (unsigned int ipart = 0; ipart < phases.size(); ++ipart) {
          phases[ipart] = (_defphases[ipart] + phasechange) % 70;
        }
      }
    }
  }

  if (phases.size() < partnames.size()) {
    // throw exception
    throw cms::Exception("InvalidAPVCyclePhases")
        << " Inconsistent phases/partitions vector sizes: " << phases.size() << " " << partnames.size();
  }

  for (unsigned int ipart = 0; ipart < partnames.size(); ++ipart) {
    //    if(phases[ipart]>=0) {
    //      apvphases->get()[partnames[ipart]] = (phases[ipart]+phasechange)%70;
    apvphases->get()[partnames[ipart]] = phases[ipart];

    //    }
  }

  iEvent.put(std::move(apvphases));
}

bool APVCyclePhaseProducerFromL1TS::isBadRun(const unsigned int run) const {
  for (std::vector<std::pair<unsigned int, unsigned int> >::const_iterator runpair = m_badruns.begin();
       runpair != m_badruns.end();
       ++runpair) {
    if (run >= runpair->first && run <= runpair->second)
      return true;
  }

  return m_badRun;
}

void APVCyclePhaseProducerFromL1TS::printConfiguration(std::stringstream& ss) const {
  ss << _defpartnames.size() << " default partition names: ";
  for (std::vector<std::string>::const_iterator part = _defpartnames.begin(); part != _defpartnames.end(); ++part) {
    ss << *part << " ";
  }
  ss << std::endl;
  ss << _defphases.size() << " default phases: ";
  for (std::vector<int>::const_iterator phase = _defphases.begin(); phase != _defphases.end(); ++phase) {
    ss << *phase << " ";
  }
  ss << std::endl;
  ss << " Magic offset: " << _magicOffset << std::endl;
  ss << " use ECO: " << _useEC0 << std::endl;
  ss << " bad run: " << m_badRun << std::endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseProducerFromL1TS);
