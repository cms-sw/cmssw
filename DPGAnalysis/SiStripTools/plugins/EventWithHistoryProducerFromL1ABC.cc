// -*- C++ -*-
//
// Package:    EventWithHistoryProducerFromL1ABC
// Class:      EventWithHistoryProducerFromL1ABC
//
/**\class EventWithHistoryProducerFromL1ABC EventWithHistoryProducerFromL1ABC.cc DPGAnalysis/SiStripTools/src/EventWithHistoryProducerFromL1ABC.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jun 30 15:26:20 CET 2009
//
//

// system include files
#include <memory>
#include <map>
#include <iostream>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

//
// class decleration
//

class EventWithHistoryProducerFromL1ABC : public edm::stream::EDProducer<> {
public:
  explicit EventWithHistoryProducerFromL1ABC(const edm::ParameterSet&);
  ~EventWithHistoryProducerFromL1ABC() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<L1AcceptBunchCrossingCollection> _l1abccollectionToken;
  edm::EDGetTokenT<TCDSRecord> _tcdsRecordToken;
  const bool _forceNoOffset;
  std::map<edm::EventNumber_t, long long> _offsets;
  long long _curroffset;
  edm::EventNumber_t _curroffevent;
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
EventWithHistoryProducerFromL1ABC::EventWithHistoryProducerFromL1ABC(const edm::ParameterSet& iConfig)
    : _l1abccollectionToken(
                            mayConsume<L1AcceptBunchCrossingCollection>(iConfig.getParameter<edm::InputTag>("l1ABCCollection"))),
      _tcdsRecordToken(
                           mayConsume<TCDSRecord>(iConfig.getParameter<edm::InputTag>("tcdsRecordLabel"))),
      
      _forceNoOffset(iConfig.getUntrackedParameter<bool>("forceNoOffset", false)),
      _offsets(),
      _curroffset(0),
      _curroffevent(0) {
  if (_forceNoOffset)
    edm::LogWarning("NoOffsetComputation") << "Orbit and BX offset will NOT be computed: Be careful!";

  produces<EventWithHistory>();

  //now do what ever other initialization is needed
}

EventWithHistoryProducerFromL1ABC::~EventWithHistoryProducerFromL1ABC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EventWithHistoryProducerFromL1ABC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (iEvent.run() < 110878) {
    std::unique_ptr<EventWithHistory> pOut(new EventWithHistory(iEvent));
    iEvent.put(std::move(pOut));

  } else {
    Handle<L1AcceptBunchCrossingCollection> pIn;
    iEvent.getByToken(_l1abccollectionToken, pIn);
    Handle<TCDSRecord> tcds_pIn;
    iEvent.getByToken(_tcdsRecordToken, tcds_pIn);
    const auto& tcdsRecord = *tcds_pIn.product();
    // offset computation

    std::cout << "From TCDSRecord: runNumber " << tcdsRecord.getRunNumber() << ", event number " << tcdsRecord.getEventNumber() << ", lumisection " << tcdsRecord.getLumiSection() << std::endl;
    
    long long orbitoffset = 0;
    int bxoffset = 0;
    long long tcds_orbitoffset = 0;
    int tcds_bxoffset = 0;
    if (!_forceNoOffset) {
      for (L1AcceptBunchCrossingCollection::const_iterator l1abc = pIn->begin(); l1abc != pIn->end(); ++l1abc) {
        if (l1abc->l1AcceptOffset() == 0) {
          orbitoffset = (long long)l1abc->orbitNumber() - (long long)iEvent.orbitNumber();
          bxoffset = l1abc->bunchCrossing() - iEvent.bunchCrossing();
          std::cout << "SCAL: orbit number: " << (long long)l1abc->orbitNumber() << ", bunchcrossing: " << l1abc->bunchCrossing() << std::endl;
        }
        else { std::cout << "Found event (" << tcdsRecord.getEventNumber() << ") with l1AcceptOffset different than zero (" << l1abc->l1AcceptOffset() << ")" << std::endl;
        }
      }
    }

    //if( !_forceNoOffset) {
    //  for (auto& recIt : tcdsRecord){
    //    if (true /*l1abc->l1AcceptOffset() == 0 Thers is no acceptoffset in tcdsrecord*/) { 
    //      tcds_orbitoffset = (long long) recIt.getOrbitNr() - (long long) iEvent.orbitNumber();
    //      tcds_bxoffset = recIt.getBXID() - iEvent.bunchCrossing();
    //    }
    //  }
    //}
    tcds_orbitoffset = (long long) tcdsRecord.getOrbitNr() - (long long) iEvent.orbitNumber();
    tcds_bxoffset = tcdsRecord.getBXID() - iEvent.bunchCrossing();
    std::cout << "TCDS: orbit number: " << (long long)tcdsRecord.getOrbitNr() << ", bunchcrossing: " << tcdsRecord.getBXID() << std::endl;
    
    std::cout << "SCAL/TCDS: Orbit offset " << orbitoffset << "/" << tcds_orbitoffset << ", bunch crossing offset " << bxoffset << "/" << tcds_bxoffset << std::endl;

    std::unique_ptr<EventWithHistory> pOut(new EventWithHistory(iEvent, *pIn, orbitoffset, bxoffset));
    iEvent.put(std::move(pOut));

    // monitor offset

    long long absbxoffset = orbitoffset * 3564 + bxoffset;

    if (_offsets.empty()) {
      _curroffset = absbxoffset;
      _curroffevent = iEvent.id().event();
      _offsets[iEvent.id().event()] = absbxoffset;
    } else {
      if (_curroffset != absbxoffset || iEvent.id().event() < _curroffevent) {
        if (_curroffset != absbxoffset) {
          edm::LogInfo("AbsoluteBXOffsetChanged")
              << "Absolute BX offset changed from " << _curroffset << " to " << absbxoffset << " at orbit "
              << iEvent.orbitNumber() << " and BX " << iEvent.bunchCrossing();
          for (L1AcceptBunchCrossingCollection::const_iterator l1abc = pIn->begin(); l1abc != pIn->end(); ++l1abc) {
            edm::LogVerbatim("AbsoluteBXOffsetChanged") << *l1abc;
          }
        }

        _curroffset = absbxoffset;
        _curroffevent = iEvent.id().event();
        _offsets[iEvent.id().event()] = absbxoffset;
      }
    }
  }
}

void EventWithHistoryProducerFromL1ABC::beginRun(const edm::Run&, const edm::EventSetup&) {
  // reset offset vector

  _offsets.clear();
  edm::LogInfo("AbsoluteBXOffsetReset") << "Absolute BX offset map reset";
}

void EventWithHistoryProducerFromL1ABC::endRun(const edm::Run&, const edm::EventSetup&) {
  // summary of absolute bx offset vector

  edm::LogInfo("AbsoluteBXOffsetSummary") << "Absolute BX offset summary:";
  for (std::map<edm::EventNumber_t, long long>::const_iterator offset = _offsets.begin(); offset != _offsets.end();
       ++offset) {
    edm::LogVerbatim("AbsoluteBXOffsetSummary") << offset->first << " " << offset->second;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryProducerFromL1ABC);
