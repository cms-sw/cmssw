/** \file CSCSegmentVisualise
 *
 *  \author D. Fortin - UC Riverside
 */

#include <RecoLocalMuon/CSCSegment/test/CSCSegmentVisualise.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"

// Constructor
CSCSegmentVisualise::CSCSegmentVisualise(const edm::ParameterSet& pset) {
  filename = pset.getUntrackedParameter<std::string>("RootFileName");
  minRechitChamber = pset.getUntrackedParameter<int>("minRechitPerChamber");
  maxRechitChamber = pset.getUntrackedParameter<int>("maxRechitPerChamber");

  geomToken_ = esConsumes();
  simHitsToken_ = consumes(edm::InputTag("g4SimHits", "MuonCSCHits"));
  recHitsToken_ = consumes(edm::InputTag("csc2DRecHits"));
  segmentsToken_ = consumes(edm::InputTag("cscSegments"));

  file = new TFile(filename.c_str(), "RECREATE");

  if (file->IsOpen())
    std::cout << "file open!" << std::endl;
  else
    std::cout << "*** Error in opening file ***" << std::endl;

  idxHisto = 0;
  ievt = 0;
}

// Destructor
CSCSegmentVisualise::~CSCSegmentVisualise() {
  file->cd();

  for (int i = 0; i < idxHisto; i++) {
    hxvsz[i]->Write();
    hyvsz[i]->Write();

    hxvszSeg[i]->Write();
    hyvszSeg[i]->Write();

    hxvszSegP[i]->Write();
    hyvszSegP[i]->Write();

    hxvszE[i]->Write();
    hyvszE[i]->Write();
  }
  file->Close();
}

// The real analysis
void CSCSegmentVisualise::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  ievt++;

  if (idxHisto > 99)
    return;

  const CSCGeometry& geom_ = eventSetup.getData(geomToken_);
  const edm::PSimHitContainer& simHits = event.get(simHitsToken_);
  const CSCRecHit2DCollection& recHits = event.get(recHitsToken_);
  const CSCSegmentCollection& segments = event.get(segmentsToken_);

  std::vector<CSCDetId> chambers;
  std::vector<CSCDetId>::const_iterator chIt;

  // First, create vector of chambers with rechits
  for (CSCRecHit2DCollection::const_iterator it2 = recHits.begin(); it2 != recHits.end(); it2++) {
    bool insert = true;
    for (chIt = chambers.begin(); chIt != chambers.end(); ++chIt)
      if (((*it2).cscDetId().chamber() == (*chIt).chamber()) && ((*it2).cscDetId().station() == (*chIt).station()) &&
          ((*it2).cscDetId().ring() == (*chIt).ring()) && ((*it2).cscDetId().endcap() == (*chIt).endcap()))
        insert = false;

    if (insert)
      chambers.push_back((*it2).cscDetId().chamberId());
  }

  for (chIt = chambers.begin(); chIt != chambers.end(); ++chIt) {
    std::vector<const CSCRecHit2D*> cscRecHits;
    std::vector<const CSCSegment*> cscSegments;
    std::vector<const CSCRecHit2D*> eRecHits;
    const CSCChamber* chamber = geom_.chamber(*chIt);

    CSCRangeMapAccessor acc;
    CSCRecHit2DCollection::range range = recHits.get(acc.cscChamber(*chIt));

    for (CSCRecHit2DCollection::const_iterator rechit = range.first; rechit != range.second; rechit++) {
      cscRecHits.push_back(&(*rechit));
    }

    if (int(cscRecHits.size()) < minRechitChamber)
      continue;
    if (int(cscRecHits.size()) > maxRechitChamber)
      continue;
    if (chamber->id().ring() == 4)
      continue;  // not interested in ME1a for now...

    std::cout << "chamber hits satisfy criteria " << std::endl;

    float xmin = -10.;
    float xmax = 10.;
    float ymin = -10.;
    float ymax = 10.;

    // Fill histograms for rechits:
    int nHits = cscRecHits.size();
    for (int i = 0; i < nHits; ++i) {
      const CSCRecHit2D* rhit = cscRecHits[i];
      LocalPoint lphit = (*rhit).localPosition();
      if (i == 0) {
        xmin = lphit.x();
        ymin = lphit.y();
        xmax = lphit.x();
        ymax = lphit.y();
      } else {
        if (lphit.x() < xmin)
          xmin = lphit.x();
        if (lphit.y() < ymin)
          ymin = lphit.y();
        if (lphit.x() > xmax)
          xmax = lphit.x();
        if (lphit.y() > ymax)
          ymax = lphit.y();
      }
    }

    xmin = xmin - 5.;
    xmax = xmax + 5.;
    ymin = ymin - 5.;
    ymax = ymax + 5.;

    char a[14];
    char evt[10];

    sprintf(evt, "Event %d", ievt);

    // Create histograms on the fly

    // rechits
    sprintf(a, "h%d", idxHisto + 100);
    hxvsz[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, xmin, xmax);
    sprintf(a, "h%d", idxHisto + 200);
    hyvsz[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, ymin, ymax);

    // Hits on segment
    sprintf(a, "h%d", idxHisto + 300);
    hxvszSeg[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, xmin, xmax);
    sprintf(a, "h%d", idxHisto + 400);
    hyvszSeg[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, ymin, ymax);

    // Segment projection
    sprintf(a, "h%d", idxHisto + 500);
    hxvszSegP[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, xmin, xmax);
    sprintf(a, "h%d", idxHisto + 600);
    hyvszSegP[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, ymin, ymax);

    // recHits from electrons/delta rays
    sprintf(a, "h%d", idxHisto + 700);
    hxvszE[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, xmin, xmax);
    sprintf(a, "h%d", idxHisto + 800);
    hyvszE[idxHisto] = new TH2F(a, evt, 100, -10., 10., 100, ymin, ymax);

    std::cout << "done creating histograms" << std::endl;

    // Fill histograms for rechits:
    for (int i = 0; i < nHits; ++i) {
      const CSCRecHit2D* rhit = cscRecHits[i];
      CSCDetId id = (CSCDetId)(*rhit).cscDetId();
      const CSCLayer* csclayer = geom_.layer(id);
      LocalPoint lphit = (*rhit).localPosition();
      GlobalPoint gphit = csclayer->toGlobal(lphit);
      LocalPoint lphitChamber = chamber->toLocal(gphit);
      float xhit = lphitChamber.x();
      float yhit = lphitChamber.y();
      float zhit = lphitChamber.z();

      hxvsz[idxHisto]->Fill(zhit, xhit);
      hyvsz[idxHisto]->Fill(zhit, yhit);
    }

    std::cout << "done filling rechit histos " << std::endl;

    // Sort rechits which comes from electron hits per chamber type:

    for (CSCRecHit2DCollection::const_iterator rec_it = range.first; rec_it != range.second; rec_it++) {
      bool isElec = false;
      float r_closest = 9999;

      CSCDetId idrec = (CSCDetId)(*rec_it).cscDetId();
      LocalPoint rhitlocal = (*rec_it).localPosition();

      for (edm::PSimHitContainer::const_iterator sim_it = simHits.begin(); sim_it != simHits.end(); sim_it++) {
        CSCDetId idsim = (CSCDetId)(*sim_it).detUnitId();

        if (idrec.endcap() == idsim.endcap() && idrec.station() == idsim.station() && idrec.ring() == idsim.ring() &&
            idrec.chamber() == idsim.chamber() && idrec.layer() == idsim.layer()) {
          LocalPoint shitlocal = (*sim_it).localPosition();

          float dx2 = (rhitlocal.x() - shitlocal.x()) * (rhitlocal.x() - shitlocal.x());
          float dy2 = (rhitlocal.y() - shitlocal.y()) * (rhitlocal.y() - shitlocal.y());
          float dr2 = dx2 + dy2;
          if (dr2 < r_closest) {
            r_closest = dr2;
            if (abs((*sim_it).particleType()) != 13) {
              isElec = true;
            } else {
              isElec = false;
            }
          }
        }
      }
      if (isElec)
        eRecHits.push_back(&(*rec_it));
    }

    nHits = eRecHits.size();

    // Fill histograms for rechits from electrons:
    for (int i = 0; i < nHits; ++i) {
      const CSCRecHit2D* rhit = eRecHits[i];
      CSCDetId id = (CSCDetId)(*rhit).cscDetId();
      const CSCLayer* csclayer = geom_.layer(id);
      LocalPoint lphit = (*rhit).localPosition();
      GlobalPoint gphit = csclayer->toGlobal(lphit);
      LocalPoint lphitChamber = chamber->toLocal(gphit);
      float xhit = lphitChamber.x();
      float yhit = lphitChamber.y();
      float zhit = lphitChamber.z();

      hxvszE[idxHisto]->Fill(zhit, xhit);
      hyvszE[idxHisto]->Fill(zhit, yhit);
    }

    // Then, sort segments per chamber type as well:

    CSCSegmentCollection::range range3 = segments.get(acc.cscChamber(*chIt));
    for (CSCSegmentCollection::const_iterator segments = range3.first; segments != range3.second; segments++) {
      cscSegments.push_back(&(*segments));
    }

    // Fill histograms for segments
    int nSegs = cscSegments.size();
    for (int j = 0; j < nSegs; ++j) {
      const CSCSegment* segs = cscSegments[j];

      LocalVector vec = (*segs).localDirection();
      LocalPoint ori = (*segs).localPosition();

      double dxdz = vec.x() / vec.z();
      double dydz = vec.y() / vec.z();

      float zstart = -10.;
      float zstop = 10.;

      float step_size = fabs(zstop - zstart) / 100.;

      std::cout << "Projected Segment" << std::endl;

      for (int i = 0; i < 100; ++i) {
        float z_proj = zstart + (i * step_size);
        float x_proj = dxdz * z_proj + ori.x();
        float y_proj = dydz * z_proj + ori.y();

        std::cout << i << " " << x_proj << " " << y_proj << " " << z_proj << std::endl;

        hxvszSegP[idxHisto]->Fill(z_proj, x_proj);
        hyvszSegP[idxHisto]->Fill(z_proj, y_proj);
      }

      // Loop over hits used in segment
      const std::vector<CSCRecHit2D>& rhseg = (*segs).specificRecHits();
      std::vector<CSCRecHit2D>::const_iterator rh_i;

      for (rh_i = rhseg.begin(); rh_i != rhseg.end(); ++rh_i) {
        CSCDetId id = (CSCDetId)(*rh_i).cscDetId();
        const CSCLayer* csclayer = geom_.layer(id);
        LocalPoint lphit = (*rh_i).localPosition();
        GlobalPoint gphit = csclayer->toGlobal(lphit);
        LocalPoint lphitChamber = chamber->toLocal(gphit);
        float xhit = lphitChamber.x();
        float yhit = lphitChamber.y();
        float zhit = lphitChamber.z();

        hxvszSeg[idxHisto]->Fill(zhit, xhit);
        hyvszSeg[idxHisto]->Fill(zhit, yhit);
      }
    }

    idxHisto++;
  }
}

DEFINE_FWK_MODULE(CSCSegmentVisualise);
