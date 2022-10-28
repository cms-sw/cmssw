/******* \class DTClusAnalyzer *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTClusAnalyzer.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecClusterCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
/* C++ Headers */
#include <iostream>
#include <cmath>
using namespace std;

/* ====================================================================== */

/* Constructor */
DTClusAnalyzer::DTClusAnalyzer(const ParameterSet& pset) : _ev(0) {
  theDtGeomToken = esConsumes();
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  // the name of the clus rec hits collection
  theRecClusLabel = pset.getParameter<string>("recClusLabel");
  theRecClusToken = consumes(edm::InputTag(theRecClusLabel));

  // the name of the 1D rec hits collection
  theRecHits1DLabel = pset.getParameter<string>("recHits1DLabel");
  theRecHits1DToken = consumes(edm::InputTag(theRecHits1DLabel));

  // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");
  theRecHits2DToken = consumes(edm::InputTag(theRecHits2DLabel));

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  bool dirStat = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kTRUE);

  /// DT histos
  new TH1F("hnClus", "Num 1d clus ", 50, 0., 50.);
  new TH1F("hClusPosX", "Local Pos Cluster :x(cm)", 100, -300., 300.);
  new TH1F("hClusRms", "Width clus ", 100, 0., 20.);
  new TH1F("hClusNHits", "# Hits clus ", 30, 0., 30.);

  new TH1F("hnHit", "Num 1d Hits ", 100, 0., 100.);

  new TH1F("hnSeg", "Num 1d seg ", 50, 0., 50.);
  new TH1F("hSegPosX", "Local Pos seg :x(cm)", 100, -300., 300.);
  new TH1F("hSegRms", "Width seg ", 100, 0., 1.);
  new TH1F("hSegNHits", "# Hits seg ", 10, 0., 10.);

  new TH2F("hnClusVsSegs", "# clus vs # segs", 30, 0, 30, 30, 0, 30);
  new TH2F("hnClusVsHits", "# clus vs # hits", 100, 0, 100, 30, 0, 30);

  // per SL
  new TH1F("hnHitSL", "Num 1d hits per SL", 100, 0., 100.);
  new TH1F("hnClusSL", "Num 1d clus per SL", 10, 0., 10.);
  new TH1F("hnSegSL", "Num 2d seg per SL", 10, 0., 10.);

  new TH2F("hnClusVsSegsSL", "# clus vs # segs per SL", 30, 0, 30, 30, 0, 30);
  new TH2F("hnClusVsHitsSL", "# clus vs # hits per SL", 100, 0, 100, 30, 0, 30);

  new TH1F("hClusSegDistSL", "#Delta x (clus segs) per SL", 100, -20, 20);
  new TH2F("hClusVsSegPosSL", "X (clus vs segs) per SL", 100, -300, 300, 100, -300, 300);

  new TH2F("hClusVsSegHitSL", "#hits (clus vs segs) per SL", 20, 0, 20, 20, 0, 20);
  TH1::AddDirectory(dirStat);
}

/* Destructor */
DTClusAnalyzer::~DTClusAnalyzer() {
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

/* Operations */
void DTClusAnalyzer::analyze(const Event& event, const EventSetup& eventSetup) {
  _ev++;

  static int j = 1;
  if ((_ev % j) == 0) {
    if ((_ev / j) == 9)
      j *= 10;
    cout << "Run:Event analyzed " << event.id().run() << ":" << event.id().event() << " Num " << _ev << endl;
  }

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom = eventSetup.getHandle(theDtGeomToken);

  // Get the 1D clusters from the event --------------
  Handle<DTRecClusterCollection> dtClusters = event.getHandle(theRecClusToken);

  // Get the 1D rechits from the event --------------
  Handle<DTRecHitCollection> dtRecHits = event.getHandle(theRecHits1DToken);

  // Get the 2D rechit collection from the event -------------------
  edm::Handle<DTRecSegment2DCollection> segs2d = event.getHandle(theRecHits2DToken);

  // only clusters
  int nClus = dtClusters->size();
  histo("hnClus")->Fill(nClus);
  for (DTRecClusterCollection::const_iterator clus = dtClusters->begin(); clus != dtClusters->end(); ++clus) {
    histo("hClusPosX")->Fill((*clus).localPosition().x());
    histo("hClusRms")->Fill(sqrt((*clus).localPositionError().xx()));
    histo("hClusNHits")->Fill((*clus).nHits());
  }

  // only segments
  int nSeg = segs2d->size();
  histo("hnSeg")->Fill(nSeg);
  for (DTRecSegment2DCollection::const_iterator seg = segs2d->begin(); seg != segs2d->end(); ++seg) {
    histo("hSegPosX")->Fill((*seg).localPosition().x());
    histo("hSegRms")->Fill(sqrt((*seg).localPositionError().xx()));
    histo("hSegNHits")->Fill((*seg).recHits().size());
  }

  int nHit = dtRecHits->size();
  histo("hnHit")->Fill(nHit);

  // clus vs segs, hits and all this
  histo2d("hnClusVsSegs")->Fill(nSeg, nClus);
  histo2d("hnClusVsHits")->Fill(nHit, nClus);

  // loop over SL and get hits, clus and segs2d for each
  const std::vector<const DTSuperLayer*>& sls = dtGeom->superLayers();
  for (auto sl = sls.begin(); sl != sls.end(); ++sl) {
    DTSuperLayerId slid((*sl)->id());

    // Hits 1d in this sl
    // SL: apparently I have to loop over the 4 layers!! What the Fuck!
    int nHitSL = 0;
    for (int il = 1; il <= 4; ++il) {
      DTRecHitCollection::range hitSL = dtRecHits->get(DTLayerId(slid, il));
      nHitSL += hitSL.second - hitSL.first;
    }

    // Cluster 1d in this sl
    DTRecClusterCollection::range clusSL = dtClusters->get(slid);
    int nClusSL = clusSL.second - clusSL.first;

    // Segment 2d in this sl
    DTRecSegment2DCollection::range segSL = segs2d->get(slid);
    int nSegSL = segSL.second - segSL.first;

    histo("hnHitSL")->Fill(nHitSL);
    histo("hnClusSL")->Fill(nClusSL);
    histo("hnSegSL")->Fill(nSegSL);
    // clus vs segs, hits and all this
    histo2d("hnClusVsSegsSL")->Fill(nSegSL, nClusSL);
    histo2d("hnClusVsHitsSL")->Fill(nHitSL, nClusSL);

    for (DTRecClusterCollection::const_iterator clus = clusSL.first; clus != clusSL.second; ++clus) {
      float minDist = 99999.;
      LocalPoint clusPos = (*clus).localPosition();
      DTRecSegment2DCollection::const_iterator closestSeg = segSL.second;
      for (DTRecSegment2DCollection::const_iterator seg = segSL.first; seg != segSL.second; ++seg) {
        LocalPoint segPos = (*seg).localPosition();
        float dist = (clusPos - segPos).mag2();
        if (dist < minDist) {
          minDist = dist;
          closestSeg = seg;
        }
      }
      if (closestSeg != segSL.second) {
        histo("hClusSegDistSL")->Fill((clusPos - (*closestSeg).localPosition()).x());
        histo2d("hClusVsSegPosSL")->Fill((*closestSeg).localPosition().x(), clusPos.x());
        histo2d("hClusVsSegHitSL")->Fill((*closestSeg).recHits().size(), (*clus).nHits());
      }
    }
  }
}

TH1F* DTClusAnalyzer::histo(const string& name) const {
  if (TH1F* h = dynamic_cast<TH1F*>(theFile->Get(name.c_str())))
    return h;
  else
    throw cms::Exception("DTSegAnalyzer") << " Not a TH1F " << name;
}

TH2F* DTClusAnalyzer::histo2d(const string& name) const {
  if (TH2F* h = dynamic_cast<TH2F*>(theFile->Get(name.c_str())))
    return h;
  else
    throw cms::Exception("DTSegAnalyzer") << " Not a TH2F " << name;
}

DEFINE_FWK_MODULE(DTClusAnalyzer);
