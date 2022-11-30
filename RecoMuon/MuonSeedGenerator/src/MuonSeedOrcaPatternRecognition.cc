/**
 *  See header file for a description of this class.
 *  
 *  All the code is under revision
 *
 *
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author ported by: R. Bellan - INFN Torino
 */

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedOrcaPatternRecognition.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Handle.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// C++
#include <vector>

using namespace std;

const std::string metname = "Muon|RecoMuon|MuonSeedOrcaPatternRecognition";

// Constructor
MuonSeedOrcaPatternRecognition::MuonSeedOrcaPatternRecognition(const edm::ParameterSet& pset,
                                                               edm::ConsumesCollector& iC)
    : MuonSeedVPatternRecognition(pset),
      theCrackEtas(pset.getParameter<std::vector<double> >("crackEtas")),
      theCrackWindow(pset.getParameter<double>("crackWindow")),
      theDeltaPhiWindow(
          pset.existsAs<double>("deltaPhiSearchWindow") ? pset.getParameter<double>("deltaPhiSearchWindow") : 0.25),
      theDeltaEtaWindow(
          pset.existsAs<double>("deltaEtaSearchWindow") ? pset.getParameter<double>("deltaEtaSearchWindow") : 0.2),
      theDeltaCrackWindow(pset.existsAs<double>("deltaEtaCrackSearchWindow")
                              ? pset.getParameter<double>("deltaEtaCrackSearchWindow")
                              : 0.25),
      muonLayersToken(iC.esConsumes<MuonDetLayerGeometry, MuonRecoGeometryRecord>()) {
  muonMeasurements = new MuonDetLayerMeasurements(theDTRecSegmentLabel.label(),
                                                  theCSCRecSegmentLabel,
                                                  edm::InputTag(),
                                                  edm::InputTag(),
                                                  theME0RecSegmentLabel,
                                                  iC,
                                                  enableDTMeasurement,
                                                  enableCSCMeasurement,
                                                  false,
                                                  false,
                                                  enableME0Measurement);
}

// reconstruct muon's seeds
void MuonSeedOrcaPatternRecognition::produce(const edm::Event& event,
                                             const edm::EventSetup& eSetup,
                                             std::vector<MuonRecHitContainer>& result) {
  // divide the RecHits by DetLayer, in order to fill the
  // RecHitContainer like it was in ORCA

  // Muon Geometry - DT, CSC, RPC and ME0
  edm::ESHandle<MuonDetLayerGeometry> muonLayers = eSetup.getHandle(muonLayersToken);

  // get the DT layers
  vector<const DetLayer*> dtLayers = muonLayers->allDTLayers();

  // get the CSC layers
  vector<const DetLayer*> cscForwardLayers = muonLayers->forwardCSCLayers();
  vector<const DetLayer*> cscBackwardLayers = muonLayers->backwardCSCLayers();

  // get the ME0 layers
  vector<const DetLayer*> me0ForwardLayers = muonLayers->forwardME0Layers();
  vector<const DetLayer*> me0BackwardLayers = muonLayers->backwardME0Layers();

  // Backward (z<0) EndCap disk
  const DetLayer* ME4Bwd = cscBackwardLayers[4];
  const DetLayer* ME3Bwd = cscBackwardLayers[3];
  const DetLayer* ME2Bwd = cscBackwardLayers[2];
  const DetLayer* ME12Bwd = cscBackwardLayers[1];
  const DetLayer* ME11Bwd = cscBackwardLayers[0];

  // Forward (z>0) EndCap disk
  const DetLayer* ME11Fwd = cscForwardLayers[0];
  const DetLayer* ME12Fwd = cscForwardLayers[1];
  const DetLayer* ME2Fwd = cscForwardLayers[2];
  const DetLayer* ME3Fwd = cscForwardLayers[3];
  const DetLayer* ME4Fwd = cscForwardLayers[4];

  // barrel
  const DetLayer* MB4DL = dtLayers[3];
  const DetLayer* MB3DL = dtLayers[2];
  const DetLayer* MB2DL = dtLayers[1];
  const DetLayer* MB1DL = dtLayers[0];

  // instantiate the accessor
  // Don not use RPC for seeding

  double barreldThetaCut = 0.2;
  // still lose good muons to a tighter cut
  double endcapdThetaCut = 1.0;

  MuonRecHitContainer list9 = filterSegments(muonMeasurements->recHits(MB4DL, event), barreldThetaCut);
  MuonRecHitContainer list6 = filterSegments(muonMeasurements->recHits(MB3DL, event), barreldThetaCut);
  MuonRecHitContainer list7 = filterSegments(muonMeasurements->recHits(MB2DL, event), barreldThetaCut);
  MuonRecHitContainer list8 = filterSegments(muonMeasurements->recHits(MB1DL, event), barreldThetaCut);

  dumpLayer("MB4 ", list9);
  dumpLayer("MB3 ", list6);
  dumpLayer("MB2 ", list7);
  dumpLayer("MB1 ", list8);

  bool* MB1 = zero(list8.size());
  bool* MB2 = zero(list7.size());
  bool* MB3 = zero(list6.size());

  MuonRecHitContainer muRH_ME0Fwd, muRH_ME0Bwd;

  if (!me0ForwardLayers.empty()) {  // Forward (z>0) EndCap disk
    const DetLayer* ME0Fwd = me0ForwardLayers[0];
    muRH_ME0Fwd = filterSegments(muonMeasurements->recHits(ME0Fwd, event), endcapdThetaCut);
  }
  if (!me0BackwardLayers.empty()) {  // Backward (z<0) EndCap disk
    const DetLayer* ME0Bwd = me0BackwardLayers[0];
    muRH_ME0Bwd = filterSegments(muonMeasurements->recHits(ME0Bwd, event), endcapdThetaCut);
  }

  endcapPatterns(filterSegments(muonMeasurements->recHits(ME11Bwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME12Bwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME2Bwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME3Bwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME4Bwd, event), endcapdThetaCut),
                 muRH_ME0Bwd,
                 list8,
                 list7,
                 list6,
                 MB1,
                 MB2,
                 MB3,
                 result);

  endcapPatterns(filterSegments(muonMeasurements->recHits(ME11Fwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME12Fwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME2Fwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME3Fwd, event), endcapdThetaCut),
                 filterSegments(muonMeasurements->recHits(ME4Fwd, event), endcapdThetaCut),
                 muRH_ME0Fwd,
                 list8,
                 list7,
                 list6,
                 MB1,
                 MB2,
                 MB3,
                 result);

  // ----------    Barrel only

  unsigned int counter = 0;
  if (list9.size() < 100) {  // +v
    for (MuonRecHitContainer::iterator iter = list9.begin(); iter != list9.end(); iter++) {
      MuonRecHitContainer seedSegments;
      seedSegments.push_back(*iter);
      complete(seedSegments, list6, MB3);
      complete(seedSegments, list7, MB2);
      complete(seedSegments, list8, MB1);
      if (check(seedSegments))
        result.push_back(seedSegments);
    }
  }

  if (list6.size() < 100) {  // +v
    for (counter = 0; counter < list6.size(); counter++) {
      if (!MB3[counter]) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list6[counter]);
        complete(seedSegments, list7, MB2);
        complete(seedSegments, list8, MB1);
        complete(seedSegments, list9);
        if (check(seedSegments))
          result.push_back(seedSegments);
      }
    }
  }

  if (list7.size() < 100) {  // +v
    for (counter = 0; counter < list7.size(); counter++) {
      if (!MB2[counter]) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list7[counter]);
        complete(seedSegments, list8, MB1);
        complete(seedSegments, list9);
        complete(seedSegments, list6, MB3);
        if (seedSegments.size() > 1 || (seedSegments.size() == 1 && seedSegments[0]->dimension() == 4)) {
          result.push_back(seedSegments);
        }
      }
    }
  }

  if (list8.size() < 100) {  // +v
    for (counter = 0; counter < list8.size(); counter++) {
      if (!MB1[counter]) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list8[counter]);
        complete(seedSegments, list9);
        complete(seedSegments, list6, MB3);
        complete(seedSegments, list7, MB2);
        if (seedSegments.size() > 1 || (seedSegments.size() == 1 && seedSegments[0]->dimension() == 4)) {
          result.push_back(seedSegments);
        }
      }
    }
  }

  if (MB3)
    delete[] MB3;
  if (MB2)
    delete[] MB2;
  if (MB1)
    delete[] MB1;

  if (result.empty()) {
    // be a little stricter with single segment seeds
    barreldThetaCut = 0.2;
    endcapdThetaCut = 0.2;

    MuonRecHitContainer all = muonMeasurements->recHits(ME4Bwd, event);
    MuonRecHitContainer tmp = filterSegments(muonMeasurements->recHits(ME3Bwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME2Bwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME12Bwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME11Bwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    if (!me0BackwardLayers.empty()) {
      const DetLayer* ME0Bwd = me0BackwardLayers[0];
      tmp = filterSegments(muonMeasurements->recHits(ME0Bwd, event), endcapdThetaCut);
      copy(tmp.begin(), tmp.end(), back_inserter(all));
    }
    if (!me0ForwardLayers.empty()) {
      const DetLayer* ME0Fwd = me0ForwardLayers[0];
      tmp = filterSegments(muonMeasurements->recHits(ME0Fwd, event), endcapdThetaCut);
      copy(tmp.begin(), tmp.end(), back_inserter(all));
    }

    tmp = filterSegments(muonMeasurements->recHits(ME11Fwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME12Fwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME2Fwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME3Fwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(ME4Fwd, event), endcapdThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(MB4DL, event), barreldThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(MB3DL, event), barreldThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(MB2DL, event), barreldThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    tmp = filterSegments(muonMeasurements->recHits(MB1DL, event), barreldThetaCut);
    copy(tmp.begin(), tmp.end(), back_inserter(all));

    LogTrace(metname) << "Number of segments: " << all.size();

    for (MuonRecHitContainer::const_iterator segmentItr = all.begin(); segmentItr != all.end(); ++segmentItr) {
      MuonRecHitContainer singleSegmentContainer;
      singleSegmentContainer.push_back(*segmentItr);
      result.push_back(singleSegmentContainer);
    }
  }
}

bool* MuonSeedOrcaPatternRecognition::zero(unsigned listSize) {
  bool* result = nullptr;
  if (listSize) {
    result = new bool[listSize];
    for (size_t i = 0; i < listSize; i++)
      result[i] = false;
  }
  return result;
}

void MuonSeedOrcaPatternRecognition::endcapPatterns(const MuonRecHitContainer& me11,
                                                    const MuonRecHitContainer& me12,
                                                    const MuonRecHitContainer& me2,
                                                    const MuonRecHitContainer& me3,
                                                    const MuonRecHitContainer& me4,
                                                    const MuonRecHitContainer& me0,
                                                    const MuonRecHitContainer& mb1,
                                                    const MuonRecHitContainer& mb2,
                                                    const MuonRecHitContainer& mb3,
                                                    bool* MB1,
                                                    bool* MB2,
                                                    bool* MB3,
                                                    std::vector<MuonRecHitContainer>& result) {
  dumpLayer("ME4 ", me4);
  dumpLayer("ME3 ", me3);
  dumpLayer("ME2 ", me2);
  dumpLayer("ME12 ", me12);
  dumpLayer("ME11 ", me11);
  dumpLayer("ME0 ", me0);

  std::vector<MuonRecHitContainer> patterns;
  MuonRecHitContainer crackSegments;
  rememberCrackSegments(me11, crackSegments);
  rememberCrackSegments(me12, crackSegments);
  rememberCrackSegments(me2, crackSegments);
  rememberCrackSegments(me3, crackSegments);
  rememberCrackSegments(me4, crackSegments);
  rememberCrackSegments(me0, crackSegments);

  const MuonRecHitContainer& list24 = me4;
  const MuonRecHitContainer& list23 = me3;

  const MuonRecHitContainer& list12 = me2;

  const MuonRecHitContainer& list22 = me12;
  MuonRecHitContainer list21 = me11;
  // add ME0 to ME1
  list21.reserve(list21.size() + me0.size());
  copy(me0.begin(), me0.end(), back_inserter(list21));

  MuonRecHitContainer list11 = list21;
  MuonRecHitContainer list5 = list22;
  MuonRecHitContainer list13 = list23;
  MuonRecHitContainer list4 = list24;

  if (list21.empty()) {
    list11 = list22;
    list5 = list21;
  }

  if (list24.size() < list23.size() && !list24.empty()) {
    list13 = list24;
    list4 = list23;
  }

  if (list23.empty()) {
    list13 = list24;
    list4 = list23;
  }

  MuonRecHitContainer list1 = list11;
  MuonRecHitContainer list2 = list12;
  MuonRecHitContainer list3 = list13;

  if (list12.empty()) {
    list3 = list12;
    if (list11.size() <= list13.size() && !list11.empty()) {
      list1 = list11;
      list2 = list13;
    } else {
      list1 = list13;
      list2 = list11;
    }
  }

  if (list13.empty()) {
    if (list11.size() <= list12.size() && !list11.empty()) {
      list1 = list11;
      list2 = list12;
    } else {
      list1 = list12;
      list2 = list11;
    }
  }

  if (!list12.empty() && !list13.empty()) {
    if (list11.size() <= list12.size() && list11.size() <= list13.size() && !list11.empty()) {  // ME 1
      if (list12.size() > list13.size()) {
        list2 = list13;
        list3 = list12;
      }
    } else if (list12.size() <= list13.size()) {  //  start with ME 2
      list1 = list12;
      if (list11.size() <= list13.size() && !list11.empty()) {
        list2 = list11;
        list3 = list13;
      } else {
        list2 = list13;
        list3 = list11;
      }
    } else {  //  start with ME 3
      list1 = list13;
      if (list11.size() <= list12.size() && !list11.empty()) {
        list2 = list11;
        list3 = list12;
      } else {
        list2 = list12;
        list3 = list11;
      }
    }
  }

  bool* ME2 = zero(list2.size());
  bool* ME3 = zero(list3.size());
  bool* ME4 = zero(list4.size());
  bool* ME5 = zero(list5.size());

  // creates list of compatible track segments
  for (MuonRecHitContainer::iterator iter = list1.begin(); iter != list1.end(); iter++) {
    if ((*iter)->recHits().size() < 4 && !list3.empty())
      continue;  // 3p.tr-seg. are not so good for starting

    MuonRecHitContainer seedSegments;
    seedSegments.push_back(*iter);
    complete(seedSegments, list2, ME2);
    complete(seedSegments, list3, ME3);
    complete(seedSegments, list4, ME4);
    complete(seedSegments, list5, ME5);
    complete(seedSegments, mb3, MB3);
    complete(seedSegments, mb2, MB2);
    complete(seedSegments, mb1, MB1);
    if (check(seedSegments))
      patterns.push_back(seedSegments);
  }

  unsigned int counter;

  for (counter = 0; counter < list2.size(); counter++) {
    if (!ME2[counter]) {
      MuonRecHitContainer seedSegments;
      seedSegments.push_back(list2[counter]);
      complete(seedSegments, list3, ME3);
      complete(seedSegments, list4, ME4);
      complete(seedSegments, list5, ME5);
      complete(seedSegments, mb3, MB3);
      complete(seedSegments, mb2, MB2);
      complete(seedSegments, mb1, MB1);
      if (check(seedSegments))
        patterns.push_back(seedSegments);
    }
  }

  if (list3.size() < 20) {  // +v
    for (counter = 0; counter < list3.size(); counter++) {
      if (!ME3[counter]) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list3[counter]);
        complete(seedSegments, list4, ME4);
        complete(seedSegments, list5, ME5);
        complete(seedSegments, mb3, MB3);
        complete(seedSegments, mb2, MB2);
        complete(seedSegments, mb1, MB1);
        if (check(seedSegments))
          patterns.push_back(seedSegments);
      }
    }
  }

  if (list4.size() < 20) {  // +v
    for (counter = 0; counter < list4.size(); counter++) {
      if (!ME4[counter]) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list4[counter]);
        complete(seedSegments, list5, ME5);
        complete(seedSegments, mb3, MB3);
        complete(seedSegments, mb2, MB2);
        complete(seedSegments, mb1, MB1);
        if (check(seedSegments))
          patterns.push_back(seedSegments);
      }
    }
  }

  if (ME5)
    delete[] ME5;
  if (ME4)
    delete[] ME4;
  if (ME3)
    delete[] ME3;
  if (ME2)
    delete[] ME2;

  if (!patterns.empty()) {
    result.insert(result.end(), patterns.begin(), patterns.end());
  } else {
    if (!crackSegments.empty()) {
      // make some single-segment seeds
      for (MuonRecHitContainer::const_iterator crackSegmentItr = crackSegments.begin();
           crackSegmentItr != crackSegments.end();
           ++crackSegmentItr) {
        MuonRecHitContainer singleSegmentPattern;
        singleSegmentPattern.push_back(*crackSegmentItr);
        result.push_back(singleSegmentPattern);
      }
    }
  }
}

void MuonSeedOrcaPatternRecognition::complete(MuonRecHitContainer& seedSegments,
                                              const MuonRecHitContainer& recHits,
                                              bool* used) const {
  MuonRecHitContainer good_rhit;
  MuonPatternRecoDumper theDumper;
  //+v get all rhits compatible with the seed on dEta/dPhi Glob.
  ConstMuonRecHitPointer first = seedSegments[0];  // first rechit of seed
  GlobalPoint ptg2 = first->globalPosition();      // its global pos +v
  for (unsigned nr = 0; nr < recHits.size(); ++nr) {
    const MuonRecHitPointer& recHit(recHits[nr]);
    GlobalPoint ptg1(recHit->globalPosition());
    float deta = fabs(ptg1.eta() - ptg2.eta());
    // Geom::Phi should keep it in the range [-pi, pi]
    float dphi = fabs(deltaPhi(ptg1.barePhi(), ptg2.barePhi()));
    // be a little more lenient in cracks
    bool crack = isCrack(recHit) || isCrack(first);
    //float detaWindow = 0.3;
    float detaWindow = crack ? theDeltaCrackWindow : theDeltaEtaWindow;
    if (deta > detaWindow || dphi > theDeltaPhiWindow) {
      continue;
    }  // +vvp!!!

    good_rhit.push_back(recHit);
    if (used)
      markAsUsed(nr, recHits, used);
  }  // recHits iter

  // select the best rhit among the compatible ones (based on Dphi Glob & Dir)
  MuonRecHitPointer best = bestMatch(first, good_rhit);
  if (best && best->isValid())
    seedSegments.push_back(best);
}

MuonSeedOrcaPatternRecognition::MuonRecHitPointer MuonSeedOrcaPatternRecognition::bestMatch(
    const ConstMuonRecHitPointer& first, MuonRecHitContainer& good_rhit) const {
  MuonRecHitPointer best = nullptr;
  if (good_rhit.size() == 1)
    return good_rhit[0];
  double bestDiscrim = 10000.;
  for (MuonRecHitContainer::iterator iter = good_rhit.begin(); iter != good_rhit.end(); iter++) {
    double discrim = discriminator(first, *iter);
    if (discrim < bestDiscrim) {
      bestDiscrim = discrim;
      best = *iter;
    }
  }
  return best;
}

double MuonSeedOrcaPatternRecognition::discriminator(const ConstMuonRecHitPointer& first,
                                                     MuonRecHitPointer& other) const {
  GlobalPoint gp1 = first->globalPosition();
  GlobalPoint gp2 = other->globalPosition();
  GlobalVector gd1 = first->globalDirection();
  GlobalVector gd2 = other->globalDirection();
  if (first->isDT() || other->isDT()) {
    return fabs(deltaPhi(gd1.barePhi(), gd2.barePhi()));
  }

  // penalize those 3-hit segments
  int nhits = other->recHits().size();
  int penalty = std::max(nhits - 2, 1);
  float dphig = deltaPhi(gp1.barePhi(), gp2.barePhi());
  // ME1A has slanted wires, so matching theta position doesn't work well.
  if (isME1A(first) || isME1A(other)) {
    return fabs(dphig / penalty);
  }

  float dthetag = gp1.theta() - gp2.theta();
  float dphid2 = fabs(deltaPhi(gd2.barePhi(), gp2.barePhi()));
  if (dphid2 > M_PI * .5)
    dphid2 = M_PI - dphid2;  //+v
  float dthetad2 = gp2.theta() - gd2.theta();
  // for CSC, make a big chi-squared of relevant variables
  // FIXME for 100 GeV mnd above muons, this doesn't perform as well as
  // previous methods.  needs investigation.
  float chisq = ((dphig / 0.02) * (dphig / 0.02) + (dthetag / 0.003) * (dthetag / 0.003) +
                 (dphid2 / 0.06) * (dphid2 / 0.06) + (dthetad2 / 0.08) * (dthetad2 / 0.08));
  return chisq / penalty;
}

bool MuonSeedOrcaPatternRecognition::check(const MuonRecHitContainer& segments) { return (segments.size() > 1); }

void MuonSeedOrcaPatternRecognition::markAsUsed(int nr, const MuonRecHitContainer& recHits, bool* used) const {
  used[nr] = true;
  // if it's ME1A with two other segments in the container, mark the ghosts as used, too.
  if (recHits[nr]->isCSC()) {
    CSCDetId detId(recHits[nr]->geographicalId().rawId());
    if (detId.ring() == 4) {
      std::vector<unsigned> chamberHitNs;
      for (unsigned i = 0; i < recHits.size(); ++i) {
        if (recHits[i]->geographicalId().rawId() == detId.rawId()) {
          chamberHitNs.push_back(i);
        }
      }
      if (chamberHitNs.size() == 3) {
        for (unsigned i = 0; i < 3; ++i) {
          used[chamberHitNs[i]] = true;
        }
      }
    }
  }
}

bool MuonSeedOrcaPatternRecognition::isCrack(const ConstMuonRecHitPointer& segment) const {
  bool result = false;
  double absEta = fabs(segment->globalPosition().eta());
  for (std::vector<double>::const_iterator crackItr = theCrackEtas.begin(); crackItr != theCrackEtas.end();
       ++crackItr) {
    if (fabs(absEta - *crackItr) < theCrackWindow) {
      result = true;
    }
  }
  return result;
}

void MuonSeedOrcaPatternRecognition::rememberCrackSegments(const MuonRecHitContainer& segments,
                                                           MuonRecHitContainer& crackSegments) const {
  for (MuonRecHitContainer::const_iterator segmentItr = segments.begin(); segmentItr != segments.end(); ++segmentItr) {
    if ((**segmentItr).hit()->dimension() == 4 && isCrack(*segmentItr)) {
      crackSegments.push_back(*segmentItr);
    }
    // save ME0 segments if eta > 2.4, no other detectors
    if ((*segmentItr)->isME0() && std::abs((*segmentItr)->globalPosition().eta()) > 2.4) {
      crackSegments.push_back(*segmentItr);
    }
  }
}

void MuonSeedOrcaPatternRecognition::dumpLayer(const char* name, const MuonRecHitContainer& segments) const {
  MuonPatternRecoDumper theDumper;

  LogTrace(metname) << name << std::endl;
  for (MuonRecHitContainer::const_iterator segmentItr = segments.begin(); segmentItr != segments.end(); ++segmentItr) {
    LogTrace(metname) << theDumper.dumpMuonId((**segmentItr).geographicalId());
  }
}

MuonSeedOrcaPatternRecognition::MuonRecHitContainer MuonSeedOrcaPatternRecognition::filterSegments(
    const MuonRecHitContainer& segments, double dThetaCut) const {
  MuonPatternRecoDumper theDumper;
  MuonRecHitContainer result;
  for (MuonRecHitContainer::const_iterator segmentItr = segments.begin(); segmentItr != segments.end(); ++segmentItr) {
    double dtheta = (*segmentItr)->globalDirection().theta() - (*segmentItr)->globalPosition().theta();
    if ((*segmentItr)->isDT()) {
      // only apply the cut to 4D segments
      if ((*segmentItr)->dimension() == 2 || fabs(dtheta) < dThetaCut) {
        result.push_back(*segmentItr);
      } else {
        LogTrace(metname) << "Cutting segment " << theDumper.dumpMuonId((**segmentItr).geographicalId())
                          << " because dtheta = " << dtheta;
      }

    } else if ((*segmentItr)->isCSC()) {
      if (fabs(dtheta) < dThetaCut) {
        result.push_back(*segmentItr);
      } else {
        LogTrace(metname) << "Cutting segment " << theDumper.dumpMuonId((**segmentItr).geographicalId())
                          << " because dtheta = " << dtheta;
      }
    } else if ((*segmentItr)->isME0()) {
      if (fabs(dtheta) < dThetaCut) {
        result.push_back(*segmentItr);
      } else {
        LogTrace(metname) << "Cutting segment " << theDumper.dumpMuonId((**segmentItr).geographicalId())
                          << " because dtheta = " << dtheta;
      }
    }
  }
  filterOverlappingChambers(result);
  return result;
}

void MuonSeedOrcaPatternRecognition::filterOverlappingChambers(MuonRecHitContainer& segments) const {
  if (segments.empty())
    return;
  MuonPatternRecoDumper theDumper;
  // need to optimize cuts
  double dphiCut = 0.05;
  double detaCut = 0.05;
  std::vector<unsigned> toKill;
  std::vector<unsigned> me1aOverlaps;
  // loop over all segment pairs to see if there are two that match up in eta and phi
  // but from different chambers
  unsigned nseg = segments.size();
  for (unsigned i = 0; i < nseg - 1; ++i) {
    GlobalPoint pg1 = segments[i]->globalPosition();
    for (unsigned j = i + 1; j < nseg; ++j) {
      GlobalPoint pg2 = segments[j]->globalPosition();
      if (segments[i]->geographicalId().rawId() != segments[j]->geographicalId().rawId() &&
          fabs(deltaPhi(pg1.barePhi(), pg2.barePhi())) < dphiCut && fabs(pg1.eta() - pg2.eta()) < detaCut) {
        LogTrace(metname) << "OVERLAP " << theDumper.dumpMuonId(segments[i]->geographicalId()) << " "
                          << theDumper.dumpMuonId(segments[j]->geographicalId());
        // see which one is best
        toKill.push_back((countHits(segments[i]) < countHits(segments[j])) ? i : j);
        if (isME1A(segments[i])) {
          me1aOverlaps.push_back(i);
          me1aOverlaps.push_back(j);
        }
      }
    }
  }

  if (toKill.empty())
    return;

  // try to kill ghosts assigned to overlaps
  for (unsigned i = 0; i < me1aOverlaps.size(); ++i) {
    DetId detId(segments[me1aOverlaps[i]]->geographicalId());
    vector<unsigned> inSameChamber;
    for (unsigned j = 0; j < nseg; ++j) {
      if (i != j && segments[j]->geographicalId() == detId) {
        inSameChamber.push_back(j);
      }
    }
    if (inSameChamber.size() == 2) {
      toKill.push_back(inSameChamber[0]);
      toKill.push_back(inSameChamber[1]);
    }
  }
  // now kill the killable
  MuonRecHitContainer result;
  for (unsigned i = 0; i < nseg; ++i) {
    if (std::find(toKill.begin(), toKill.end(), i) == toKill.end()) {
      result.push_back(segments[i]);
    }
  }
  segments.swap(result);
}

bool MuonSeedOrcaPatternRecognition::isME1A(const ConstMuonRecHitPointer& segment) const {
  return segment->isCSC() && CSCDetId(segment->geographicalId()).ring() == 4;
}

int MuonSeedOrcaPatternRecognition::countHits(const MuonRecHitPointer& segment) const {
  int count = 0;
  vector<TrackingRecHit*> components = (*segment).recHits();
  for (vector<TrackingRecHit*>::const_iterator component = components.begin(); component != components.end();
       ++component) {
    int componentSize = (**component).recHits().size();
    count += (componentSize == 0) ? 1 : componentSize;
  }
  return count;
}
