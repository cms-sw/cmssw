/******* \class DTEffAnalyzer *******
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
#include "RecoLocalMuon/DTSegment/test/DTEffAnalyzer.h"

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

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

/* C++ Headers */
#include <iostream>
#include <cmath>
using namespace std;

/* ====================================================================== */

/* Constructor */
DTEffAnalyzer::DTEffAnalyzer(const ParameterSet& pset) : theDtGeomToken(esConsumes()) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  // the name of the 1D rec hits collection
  theRecHits1DLabel = pset.getParameter<string>("recHits1DLabel");

  // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");

  theMinHitsSegment = static_cast<unsigned int>(pset.getParameter<int>("minHitsSegment"));
  theMinChi2NormSegment = pset.getParameter<double>("minChi2NormSegment");
  theMinCloseDist = pset.getParameter<double>("minCloseDist");

  consumes<DTRecSegment4DCollection>(theRecHits4DLabel);
}

void DTEffAnalyzer::beginJob() {
  if (debug)
    cout << "beginOfJob" << endl;

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  bool dirStat = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kTRUE);

  // trigger Histos
  new TH1F("hTrigBits", "All trigger bits", 10, 0., 10.);

  for (int w = -2; w <= 2; ++w) {
    stringstream nameWheel;
    nameWheel << "_Wh" << w;
    //cout << "Wheel " << nameWheel.str() << endl;
    for (int sec = 1; sec <= 14; ++sec) {  // section 1 to 14
      stringstream nameSector;
      nameSector << nameWheel.str() << "_Sec" << sec;
      //cout << "Sec " << nameSector.str() << endl;
      for (int st = 1; st <= 4; ++st) {  // station 1 to 4

        stringstream nameChamber;
        nameChamber << nameSector.str() << "_St" << st;

        //cout << nameChamber << endl;
        createTH1F("hDistSegFromExtrap", "Distance segments from extrap position ", nameChamber.str(), 200, 0., 200.);
        createTH1F("hNaiveEffSeg", "Naive eff ", nameChamber.str(), 10, 0., 10.);
        createTH2F(
            "hEffSegVsPosDen", "Eff vs local position (all) ", nameChamber.str(), 25, -250., 250., 25, -250., 250.);
        createTH2F(
            "hEffGoodSegVsPosDen", "Eff vs local position (good) ", nameChamber.str(), 25, -250., 250., 25, -250., 250.);
        createTH2F("hEffSegVsPosNum", "Eff vs local position ", nameChamber.str(), 25, -250., 250., 25, -250., 250.);
        createTH2F("hEffGoodSegVsPosNum",
                   "Eff vs local position (good segs) ",
                   nameChamber.str(),
                   25,
                   -250.,
                   250.,
                   25,
                   -250.,
                   250.);
        createTH2F("hEffGoodCloseSegVsPosNum",
                   "Eff vs local position (good aand close segs) ",
                   nameChamber.str(),
                   25,
                   -250.,
                   250.,
                   25,
                   -250.,
                   250.);
      }
    }
  }
  // cout << "List of created histograms " << endl;
  // theFile->ls();
  TH1::AddDirectory(dirStat);
}

/* Destructor */
DTEffAnalyzer::~DTEffAnalyzer() {
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

/* Operations */
void DTEffAnalyzer::analyze(const Event& event, const EventSetup& eventSetup) {
  dtGeom = eventSetup.getHandle(theDtGeomToken);

  if (debug)
    cout << endl
         << "--- [DTEffAnalyzer] Event analysed #Run: " << event.id().run() << " #Event: " << event.id().event()
         << endl;

  effSegments(event, eventSetup);
}

void DTEffAnalyzer::effSegments(const Event& event, const EventSetup& eventSetup) {
  // Get the 4D rechit collection from the event -------------------
  // Handle<DTRecSegment4DCollection> segs;
  event.getByLabel(theRecHits4DLabel, segs);
  if (debug) {
    cout << "4d " << segs->size() << endl;
    for (DTRecSegment4DCollection::const_iterator seg = segs->begin(); seg != segs->end(); ++seg)
      cout << *seg << endl;
  }

  // Get events with 3 segments in different station and look what happen on
  // the other station. Note, must take into account geometrical acceptance

  // trivial pattern recognition: get 3 segments in 3 different station of a
  // given wheel, sector

  for (int wheel = -2; wheel <= 2; ++wheel) {
    for (int sector = 1; sector <= 12; ++sector) {
      evaluateEff(DTChamberId(wheel, 1, sector), 2, 3);  // get efficiency for MB1 using MB2 and MB3
      evaluateEff(DTChamberId(wheel, 2, sector), 1, 3);  // get efficiency for MB2 using MB1 and MB3
      evaluateEff(DTChamberId(wheel, 3, sector), 2, 4);  // get efficiency for MB3 using MB2 and MB4
      evaluateEff(DTChamberId(wheel, 4, sector), 2, 3);  // get efficiency for MB4 using MB2 and MB3
    }
  }
}

void DTEffAnalyzer::evaluateEff(const DTChamberId& MidId, int bottom, int top) const {
  if (debug)
    cout << "evaluateEff " << MidId << " bott/top " << bottom << "/" << top << endl;
  // Select events with (good) segments in Bot and Top
  DTChamberId BotId(MidId.wheel(), bottom, MidId.sector());
  DTChamberId TopId(MidId.wheel(), top, MidId.sector());

  // Get segments in the bottom chambers (if any)
  DTRecSegment4DCollection::range segsBot = segs->get(BotId);
  int nSegsBot = segsBot.second - segsBot.first;
  // check if any segments is there
  if (nSegsBot == 0)
    return;

  // Get segments in the top chambers (if any)
  DTRecSegment4DCollection::range segsTop = segs->get(TopId);
  int nSegsTop = segsTop.second - segsTop.first;

  // something more sophisticate check quality of segments
  const DTRecSegment4D& bestBotSeg = getBestSegment(segsBot);
  //cout << "BestBotSeg " << bestBotSeg << endl;

  DTRecSegment4D* pBestTopSeg = 0;
  if (nSegsTop > 0)
    pBestTopSeg = const_cast<DTRecSegment4D*>(&getBestSegment(segsTop));
  //if top chamber is MB4 sector 10, consider also sector 14
  if (TopId.station() == 4 && TopId.sector() == 10) {
    // cout << "nSegsTop " << nSegsTop << endl;
    // cout << "pBestTopSeg " << pBestTopSeg << endl;
    // cout << "Ch " << TopId << endl;
    DTChamberId TopId14(MidId.wheel(), top, 14);
    DTRecSegment4DCollection::range segsTop14 = segs->get(TopId14);
    int nSegsTop14 = segsTop14.second - segsTop14.first;
    //cout << "nSegsTop14 " << nSegsTop14 << endl;
    nSegsTop += nSegsTop;
    if (nSegsTop14) {
      DTRecSegment4D* pBestTopSeg14 = const_cast<DTRecSegment4D*>(&getBestSegment(segsTop14));

      // get best between sector 10 and 14
      pBestTopSeg = const_cast<DTRecSegment4D*>(getBestSegment(pBestTopSeg, pBestTopSeg14));
      //cout << "pBestTopSeg " << pBestTopSeg << endl;
    }
  }
  if (!pBestTopSeg)
    return;
  const DTRecSegment4D& bestTopSeg = *pBestTopSeg;
  //cout << "BestTopSeg " << bestTopSeg << endl;

  DTRecSegment4DCollection::range segsMid = segs->get(MidId);
  int nSegsMid = segsMid.second - segsMid.first;
  //cout << "nSegsMid " << nSegsMid << endl;

  // very trivial efficiency, just count segments
  // cout << "MidId " << MidId << endl;
  // cout << "histo " << hName("hNaiveEffSeg",MidId) << endl;
  // cout << histo(hName("hNaiveEffSeg",MidId)) << endl;
  histo(hName("hNaiveEffSeg", MidId))->Fill(0);
  if (nSegsMid > 0)
    histo(hName("hNaiveEffSeg", MidId))->Fill(1);

  // get position at Mid by interpolating the position (not direction) of best
  // segment in Bot and Top to Mid surface
  LocalPoint posAtMid = interpolate(bestBotSeg, bestTopSeg, MidId);
  // cout << "PosAtMid " << posAtMid << endl;

  // is best segment good enough?
  //cout << "about to good " << endl;
  if (isGoodSegment(bestBotSeg) && isGoodSegment(bestTopSeg)) {
    histo2d(hName("hEffSegVsPosDen", MidId))->Fill(posAtMid.x(), posAtMid.y());
    //check if interpolation fall inside middle chamber
    if ((dtGeom->chamber(MidId))->surface().bounds().inside(posAtMid)) {
      // cout << "IsInside" << endl;

      //cout << "good" << endl;
      histo2d(hName("hEffGoodSegVsPosDen", MidId))->Fill(posAtMid.x(), posAtMid.y());

      if (nSegsMid > 0) {
        histo(hName("hNaiveEffSeg", MidId))->Fill(2);
        histo2d(hName("hEffSegVsPosNum", MidId))->Fill(posAtMid.x(), posAtMid.y());
        const DTRecSegment4D& bestMidSeg = getBestSegment(segsMid);
        // check if middle segments is good enough
        if (isGoodSegment(bestMidSeg)) {
          histo2d(hName("hEffGoodSegVsPosNum", MidId))->Fill(posAtMid.x(), posAtMid.y());
          LocalPoint midSegPos = bestMidSeg.localPosition();
          // check if middle segments is also close enough
          double dist;
          // cout << "bestBotSeg " << bestBotSeg.hasPhi() << " " <<
          //    bestBotSeg.hasZed() << " " << bestBotSeg << endl;
          // cout << "bestTopSeg " << bestTopSeg.hasPhi() << " " <<
          //   bestTopSeg.hasZed() << " " << bestTopSeg << endl;
          // cout << "midSegPos " << midSegPos << endl;
          // cout << "posAtMid " << posAtMid<< endl;
          // cout << "bestMidSeg " << bestMidSeg.hasPhi() << " " <<
          //   bestMidSeg.hasZed() << " " << bestMidSeg << endl;
          if (bestMidSeg.hasPhi()) {
            if (bestTopSeg.hasZed() && bestBotSeg.hasZed() && bestMidSeg.hasZed()) {
              dist = (midSegPos - posAtMid).mag();
            } else {
              dist = fabs((midSegPos - posAtMid).x());
            }
          } else {
            dist = fabs((midSegPos - posAtMid).y());
          }
          // cout << "dist " << dist << " theMinCloseDist " << theMinCloseDist<< endl;
          if (dist < theMinCloseDist) {
            histo2d(hName("hEffGoodCloseSegVsPosNum", MidId))->Fill(posAtMid.x(), posAtMid.y());
          }
          histo(hName("hDistSegFromExtrap", MidId))->Fill(dist);
        }
      }
    }
  }
  // else cout << "Outside " << endl;
}

// as usual max number of hits and min chi2
const DTRecSegment4D& DTEffAnalyzer::getBestSegment(const DTRecSegment4DCollection::range& segs) const {
  DTRecSegment4DCollection::const_iterator bestIter;
  unsigned int nHitBest = 0;
  double chi2Best = 99999.;
  for (DTRecSegment4DCollection::const_iterator seg = segs.first; seg != segs.second; ++seg) {
    unsigned int nHits = ((*seg).hasPhi() ? (*seg).phiSegment()->recHits().size() : 0);
    nHits += ((*seg).hasZed() ? (*seg).zSegment()->recHits().size() : 0);

    if (nHits == nHitBest) {
      if ((*seg).chi2() / (*seg).degreesOfFreedom() < chi2Best) {
        chi2Best = (*seg).chi2() / (*seg).degreesOfFreedom();
        bestIter = seg;
      }
    } else if (nHits > nHitBest) {
      nHitBest = nHits;
      bestIter = seg;
    }
  }
  return *bestIter;
}

const DTRecSegment4D* DTEffAnalyzer::getBestSegment(const DTRecSegment4D* s1, const DTRecSegment4D* s2) const {
  if (!s1)
    return s2;
  if (!s2)
    return s1;
  unsigned int nHits1 = (s1->hasPhi() ? s1->phiSegment()->recHits().size() : 0);
  nHits1 += (s1->hasZed() ? s1->zSegment()->recHits().size() : 0);

  unsigned int nHits2 = (s2->hasPhi() ? s2->phiSegment()->recHits().size() : 0);
  nHits2 += (s2->hasZed() ? s2->zSegment()->recHits().size() : 0);

  if (nHits1 == nHits2) {
    if (s1->chi2() / s1->degreesOfFreedom() < s2->chi2() / s2->degreesOfFreedom())
      return s1;
    else
      return s2;
  } else if (nHits1 > nHits2)
    return s1;
  return s2;
}

bool DTEffAnalyzer::isGoodSegment(const DTRecSegment4D& seg) const {
  if (seg.chamberId().station() != 4 && !seg.hasZed())
    return false;
  unsigned int nHits = (seg.hasPhi() ? seg.phiSegment()->recHits().size() : 0);
  nHits += (seg.hasZed() ? seg.zSegment()->recHits().size() : 0);
  return (nHits >= theMinHitsSegment && seg.chi2() / seg.degreesOfFreedom() < theMinChi2NormSegment);
}

LocalPoint DTEffAnalyzer::interpolate(const DTRecSegment4D& seg1,
                                      const DTRecSegment4D& seg3,
                                      const DTChamberId& id2) const {
  // Get GlobalPoition of Seg in MB1
  GlobalPoint gpos1 = (dtGeom->chamber(seg1.chamberId()))->toGlobal(seg1.localPosition());

  // Get GlobalPoition of Seg in MB3
  GlobalPoint gpos3 = (dtGeom->chamber(seg3.chamberId()))->toGlobal(seg3.localPosition());

  // interpolate
  // get all in MB2 frame
  LocalPoint pos1 = (dtGeom->chamber(id2))->toLocal(gpos1);
  LocalPoint pos3 = (dtGeom->chamber(id2))->toLocal(gpos3);
  // cout << "pos1 " << pos1 << endl;
  // cout << "pos3 " << pos3 << endl;

  // case 1: 1 and 3 has both projection. No problem

  // case 2: one projection is missing for one of the segments. Keep the other's
  // segment position
  if (!seg1.hasZed())
    pos1 = LocalPoint(pos1.x(), pos3.y(), pos1.z());
  if (!seg3.hasZed())
    pos3 = LocalPoint(pos3.x(), pos1.y(), pos3.z());

  if (!seg1.hasPhi())
    pos1 = LocalPoint(pos3.x(), pos1.y(), pos1.z());
  if (!seg3.hasPhi())
    pos3 = LocalPoint(pos1.x(), pos3.y(), pos3.z());

  // cout << "pos1 " << pos1 << endl;
  // cout << "pos3 " << pos3 << endl;
  // direction
  LocalVector dir = (pos3 - pos1).unit();  // z points inward!
  // cout << "dir " << dir << endl;
  LocalPoint pos2 = pos1 + dir * pos1.z() / (-dir.z());
  // cout << "pos2 " << pos2 << endl;

  return pos2;
}

TH1F* DTEffAnalyzer::histo(const string& name) const {
  if (!theFile->Get(name.c_str()))
    throw cms::Exception("DTEffAnalyzer") << " TH1F not existing " << name;
  if (TH1F* h = dynamic_cast<TH1F*>(theFile->Get(name.c_str())))
    return h;
  else
    throw cms::Exception("DTEffAnalyzer") << " Not a TH1F " << name;
}

TH2F* DTEffAnalyzer::histo2d(const string& name) const {
  if (!theFile->Get(name.c_str()))
    throw cms::Exception("DTEffAnalyzer") << " TH1F not existing " << name;
  if (TH2F* h = dynamic_cast<TH2F*>(theFile->Get(name.c_str())))
    return h;
  else
    throw cms::Exception("DTEffAnalyzer") << " Not a TH2F " << name;
}

string DTEffAnalyzer::toString(const DTChamberId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() << "_Sec" << id.sector() << "_St" << id.station();
  return result.str();
}

template <class T>
string DTEffAnalyzer::hName(const string& s, const T& id) const {
  string name(toString(id));
  stringstream hName;
  hName << s << name;
  return hName.str();
}

void DTEffAnalyzer::createTH1F(const std::string& name,
                               const std::string& title,
                               const std::string& suffix,
                               int nbin,
                               const double& binMin,
                               const double& binMax) const {
  stringstream hName;
  stringstream hTitle;
  hName << name << suffix;
  hTitle << title << suffix;
  new TH1F(hName.str().c_str(), hTitle.str().c_str(), nbin, binMin, binMax);
}

void DTEffAnalyzer::createTH2F(const std::string& name,
                               const std::string& title,
                               const std::string& suffix,
                               int nBinX,
                               const double& binXMin,
                               const double& binXMax,
                               int nBinY,
                               const double& binYMin,
                               const double& binYMax) const {
  stringstream hName;
  stringstream hTitle;
  hName << name << suffix;
  hTitle << title << suffix;
  new TH2F(hName.str().c_str(), hTitle.str().c_str(), nBinX, binXMin, binXMax, nBinY, binYMin, binYMax);
}

DEFINE_FWK_MODULE(DTEffAnalyzer);
