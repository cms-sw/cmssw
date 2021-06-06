#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>

#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include <sstream>
#include <cstdlib>

CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset,
                                     bool TMB07,
                                     const L1MuTriggerScales* scales,
                                     const L1MuTriggerPtScale* ptScale) {
  m_minBX = pset.getParameter<int>("MinBX");
  m_maxBX = pset.getParameter<int>("MaxBX");

  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e) {
    for (int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s) {
      // All SPs work with the same configuration (impossible to make it more exclusive in this framework)
      my_SPs[e - 1][s - 1] = new CSCTFSectorProcessor(e, s, pset, TMB07, scales, ptScale);
    }
  }
}

void CSCTFTrackBuilder::initialize(const edm::EventSetup& c, const Tokens& tokens) {
  //my_dtrc->initialize(c);
  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e) {
    for (int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s) {
      my_SPs[e - 1][s - 1]->initialize(c, tokens);
    }
  }
}

CSCTFTrackBuilder::~CSCTFTrackBuilder() {
  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e) {
    for (int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s) {
      delete my_SPs[e - 1][s - 1];
      my_SPs[e - 1][s - 1] = nullptr;
    }
  }
}

void CSCTFTrackBuilder::buildTracks(
    const CSCCorrelatedLCTDigiCollection* lcts,
    const CSCTriggerContainer<csctf::TrackStub>* dtstubss,  //const L1MuDTChambPhContainer* dttrig,
    L1CSCTrackCollection* trkcoll,
    CSCTriggerContainer<csctf::TrackStub>* stubs_to_dt) {
  std::vector<csc::L1Track> trks;
  CSCTriggerContainer<csctf::TrackStub> stub_list;

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  for (Citer = lcts->begin(); Citer != lcts->end(); Citer++) {
    CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

    for (; Diter != Dend; Diter++) {
      csctf::TrackStub theStub((*Diter), (*Citer).first);
      stub_list.push_back(theStub);
    }
  }

  // Now we append the track stubs the the DT Sector Collector
  // after processing from the DT Receiver.

  //  CSCTriggerContainer<csctf::TrackStub> dtstubs = my_dtrc->process(dttrig);
  //  stub_list.push_many(dtstubs);
  stub_list.push_many(*dtstubss);

  // run each sector processor in the TF
  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e) {
    for (int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s) {
      CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e, s);
      int spReturnValue = my_SPs[e - 1][s - 1]->run(current_e_s);
      if (spReturnValue == -1)  //Major Error, returning with empty Coll's
      {
        trkcoll->clear();
        stubs_to_dt->clear();
        return;
      } else if (spReturnValue) {
        std::vector<csc::L1Track> theTracks = my_SPs[e - 1][s - 1]->tracks().get();
        trks.insert(trks.end(), theTracks.begin(), theTracks.end());
      }
      stubs_to_dt->push_many(my_SPs[e - 1][s - 1]->dtStubs());  // send stubs whether or not we find a track!!!
    }
  }

  // Now to combine tracks with their track stubs and send them off.
  trkcoll->resize(trks.size());
  std::vector<csc::L1Track>::const_iterator titr = trks.begin();
  L1CSCTrackCollection::iterator tcitr = trkcoll->begin();

  for (; titr != trks.end(); titr++) {
    tcitr->first = (*titr);
    std::vector<csctf::TrackStub> possible_stubs = my_SPs[titr->endcap() - 1][titr->sector() - 1]->filteredStubs();
    std::vector<csctf::TrackStub>::const_iterator tkstbs = possible_stubs.begin();

    int me1ID = titr->me1ID();
    int me2ID = titr->me2ID();
    int me3ID = titr->me3ID();
    int me4ID = titr->me4ID();
    int mb1ID = titr->mb1ID();
    int me1delay = titr->me1Tbin();
    int me2delay = titr->me2Tbin();
    int me3delay = titr->me3Tbin();
    int me4delay = titr->me4Tbin();
    int mb1delay = titr->mb1Tbin();
    // BX analyzer: some stub could be delayed by BXA so that all the stubs will run through the core at the same BX;
    //  then there is a rule of "second earlies LCT": resulting track will be placed at BX of the "second earliest LCT";
    //  in the end there are two parameters in place: the delay by BXA w.r.t to the last LCT and track tbin assignment
    std::map<int, std::list<int> > timeline;
    if (me1ID)
      timeline[me1delay].push_back(1);
    if (me2ID)
      timeline[me2delay].push_back(2);
    if (me3ID)
      timeline[me3delay].push_back(3);
    if (me4ID)
      timeline[me4delay].push_back(4);
    int earliest_tbin = 0, second_earliest_tbin = 0;
    for (int bx = 7; bx >= 0; bx--) {
      std::list<int>::const_iterator iter = timeline[bx].begin();
      while (iter != timeline[bx].end()) {
        if (earliest_tbin == 0)
          earliest_tbin = bx;
        else if (second_earliest_tbin == 0)
          second_earliest_tbin = bx;
        iter++;
      }
    }
    // Core's input was loaded in a relative time window BX=[0-7)
    // To relate it to time window of tracks (centred at BX=0) we introduce a shift:
    int shift = (m_maxBX + m_minBX) / 2 - m_minBX + m_minBX;
    int me1Tbin = titr->bx() - me1delay + second_earliest_tbin + shift;
    int me2Tbin = titr->bx() - me2delay + second_earliest_tbin + shift;
    int me3Tbin = titr->bx() - me3delay + second_earliest_tbin + shift;
    int me4Tbin = titr->bx() - me4delay + second_earliest_tbin + shift;
    int mb1Tbin = titr->bx() - mb1delay + second_earliest_tbin + shift;

    for (; tkstbs != possible_stubs.end(); tkstbs++) {
      switch (tkstbs->station()) {
        case 1:
          if ((tkstbs->getMPCLink() +
               (3 * (CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(tkstbs->getDetId().rawId())) - 1))) ==
                  me1ID &&
              me1ID != 0 && me1Tbin == tkstbs->BX()) {
            tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
          }
          break;
        case 2:
          if (tkstbs->getMPCLink() == me2ID && me2ID != 0 && me2Tbin == tkstbs->BX()) {
            tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
          }
          break;
        case 3:
          if (tkstbs->getMPCLink() == me3ID && me3ID != 0 && me3Tbin == tkstbs->BX()) {
            tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
          }
          break;
        case 4:
          if (tkstbs->getMPCLink() == me4ID && me4ID != 0 && me4Tbin == tkstbs->BX()) {
            tcitr->second.insertDigi(CSCDetId(tkstbs->getDetId().rawId()), *(tkstbs->getDigi()));
          }
          break;
        case 5:
          if (tkstbs->getMPCLink() == mb1ID && mb1ID != 0 && mb1Tbin == tkstbs->BX()) {
            /// Hmmm how should I implement this??? Maybe change the L1Track to use stubs not LCTs?
          }
          break;
        default:
          edm::LogWarning("CSCTFTrackBuilder::buildTracks()")
              << "SERIOUS ERROR: STATION " << tkstbs->station() << " NOT IN RANGE [1,5]\n";
      };
    }
    tcitr++;  // increment to next track in the collection
  }
}
