/*
 *  See header file for a description of this class.
 *
 */

#include "DTOccupancyEfficiency.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <iterator>

using namespace edm;
using namespace std;

DTOccupancyEfficiency::DTOccupancyEfficiency(const ParameterSet& pset) {
  debug = pset.getUntrackedParameter<bool>("debug", false);
  // label for dtDigis
  dtDigiToken_ = consumes<DTDigiCollection>(edm::InputTag(pset.getUntrackedParameter<string>("digiLabel")));
  // the name of the 4D rec hits collection
  recHits4DToken_ =
      consumes<DTRecSegment4DCollection>(edm::InputTag(pset.getUntrackedParameter<string>("recHits4DLabel")));
  // the name of the rechits collection
  recHitToken_ = consumes<DTRecHitCollection>(edm::InputTag(pset.getUntrackedParameter<string>("recHitLabel")));

  parameters = pset;
}

DTOccupancyEfficiency::~DTOccupancyEfficiency() {}

void DTOccupancyEfficiency::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& context) {
  ibooker.setCurrentFolder("DT/04-OccupancyEfficiency/digisPerRing");
  // Digis per ring
  for (int station = 1; station < 5; station++) {
    string station_s = to_string(station);
    for (int wheel = -2; wheel < 3; wheel++) {
      string wheel_s = to_string(wheel);
      if (wheel > 0)
        wheel_s = "+" + wheel_s;
      string histoName = "digisPerMB" + station_s + "W" + wheel_s;
      string histoTitle = "Number of digis in MB" + station_s + "YB" + wheel_s;
      (digisPerRing[station])[wheel] = ibooker.book1D(histoName, histoTitle, 100, 0, 150);
    }
  }

  ibooker.setCurrentFolder("DT/04-OccupancyEfficiency/timeBoxesPerRing");
  // TimeBoxes per ring
  for (int station = 1; station < 5; station++) {
    string station_s = to_string(station);
    for (int wheel = -2; wheel < 3; wheel++) {
      string wheel_s = to_string(wheel);
      if (wheel > 0)
        wheel_s = "+" + wheel_s;
      string histoName = "timeBoxesPerMB" + station_s + "W" + wheel_s;
      string histoTitle = "Number of TDC counts in MB" + station_s + "YB" + wheel_s;
      (timeBoxesPerRing[station])[wheel] = ibooker.book1D(histoName, histoTitle, 400, 0, 1600);
    }
  }

  ibooker.setCurrentFolder("DT/04-OccupancyEfficiency");

  // TimeBoxes
  timeBoxesPerEvent = ibooker.book1D("timeBoxesPerEvent", "TDC counts per event", 400, 0, 1600);

  // Digis
  digisPerEvent = ibooker.book1D("digisPerEvent", "Number of digis per event", 100, 0, 900);

  // RecHits
  recHitsPerEvent = ibooker.book1D("recHitsPerEvent", "Number of RecHits per event", 100, 0, 250);

  // 4D segments
  segments4DPerEvent = ibooker.book1D("segments4DPerEvent", "Number of 4D Segments per event", 50, 0, 50);

  recHitsPer4DSegment = ibooker.book1D("recHitsPer4DSegment", "Number of RecHits per segment", 16, 0.5, 16.5);

  // T0 from segements
  t0From4DPhiSegment = ibooker.book1D("t0From4DPhiSegment", "T0 from 4D Phi segments", 100, -150, 150);
  t0From4DZSegment = ibooker.book1D("t0From4DZSegment", "T0 from 4D Z segments", 100, -150, 150);
}

void DTOccupancyEfficiency::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  if (debug)
    cout << "[DTOccupancyEfficiency] Analyze #Run: " << event.id().run() << " #Event: " << event.id().event() << endl;

  // Digi collection
  edm::Handle<DTDigiCollection> dtdigis;
  event.getByToken(dtDigiToken_, dtdigis);

  int numberOfDigis = 0;
  std::map<int, std::map<int, int>> numberOfDigisPerRing;

  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It = dtdigis->begin(); dtLayerId_It != dtdigis->end(); ++dtLayerId_It) {  // Loop over layers
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
         digiIt != ((*dtLayerId_It).second).second;
         ++digiIt) {
      numberOfDigis++;
      int wheelId = ((*dtLayerId_It).first).wheel();
      int stationId = ((*dtLayerId_It).first).station();
      (numberOfDigisPerRing[stationId])[wheelId] += 1;

      timeBoxesPerEvent->Fill((*digiIt).countsTDC());
      (timeBoxesPerRing[stationId])[wheelId]->Fill((*digiIt).countsTDC());
    }
  }

  // Total number of Digis per Event
  digisPerEvent->Fill(numberOfDigis);

  // Digis per Ring in every wheel
  for (int station = 1; station < 5; station++)
    for (int wheel = -2; wheel < 3; wheel++)
      (digisPerRing[station])[wheel]->Fill((numberOfDigisPerRing[station])[wheel]);

  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByToken(recHits4DToken_, all4DSegments);

  segments4DPerEvent->Fill(all4DSegments->size());

  // Get the rechit collection from the event
  Handle<DTRecHitCollection> dtRecHits;
  event.getByToken(recHitToken_, dtRecHits);

  recHitsPerEvent->Fill(dtRecHits->size());

  // Number of RecHits per segment and t0 from Segment
  // Loop on all segments
  for (DTRecSegment4DCollection::const_iterator segment = all4DSegments->begin(); segment != all4DSegments->end();
       ++segment) {
    unsigned int nHits = (segment->hasPhi() ? (segment->phiSegment())->recHits().size() : 0);
    nHits += (segment->hasZed() ? (segment->zSegment())->recHits().size() : 0);
    recHitsPer4DSegment->Fill(nHits);

    if (segment->hasPhi()) {
      double segmentPhiT0 = segment->phiSegment()->t0();
      if (segment->phiSegment()->ist0Valid())
        t0From4DPhiSegment->Fill(segmentPhiT0);
    }
    if (segment->hasZed()) {
      double segmentZT0 = segment->zSegment()->t0();
      if (segment->zSegment()->ist0Valid())
        t0From4DZSegment->Fill(segmentZT0);
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
