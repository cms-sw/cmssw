#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

DTSegmentSelector::DTSegmentSelector(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
    : muonTags_(pset.getParameter<edm::InputTag>("Muons")),
      checkNoisyChannels_(pset.getParameter<bool>("checkNoisyChannels")),
      minHitsPhi_(pset.getParameter<int>("minHitsPhi")),
      minHitsZ_(pset.getParameter<int>("minHitsZ")),
      maxChi2_(pset.getParameter<double>("maxChi2")),
      maxAnglePhi_(pset.getParameter<double>("maxAnglePhi")),
      maxAngleZ_(pset.getParameter<double>("maxAngleZ")) {
  muonToken_ = iC.consumes<reco::MuonCollection>(muonTags_);
}

bool DTSegmentSelector::operator()(DTRecSegment4D const& segment,
                                   edm::Event const& event,
                                   edm::EventSetup const& setup) {
  bool result = true;

  /* get the muon collection if one is specified
    (not specifying a muon collection switches off muon matching */
  if (!muonTags_.label().empty()) {
    edm::Handle<reco::MuonCollection> muonH;
    event.getByToken(muonToken_, muonH);
    const std::vector<reco::Muon>* muons = muonH.product();
    //std::cout << " Muon collection size: " << muons->size() << std::endl;
    if (muons->empty())
      return false;

    DTChamberId segId(segment.rawId());

    bool matched = false;
    for (auto& imuon : *muons)
      for (const auto& ch : imuon.matches()) {
        DetId chId(ch.id.rawId());
        if (chId != segId)
          continue;
        if (imuon.pt() < 15 || !imuon.isGlobalMuon())
          continue;

        int nsegs = ch.segmentMatches.size();
        if (!nsegs)
          continue;

        LocalPoint posHit = segment.localPosition();
        float dx = (posHit.x() ? posHit.x() - ch.x : 0);
        float dy = (posHit.y() ? posHit.y() - ch.y : 0);
        float dr = sqrt(dx * dx + dy * dy);
        if (dr < 5)
          matched = true;
      }

    if (!matched)
      result = false;
  }

  edm::ESHandle<DTStatusFlag> statusMap;
  if (checkNoisyChannels_)
    setup.get<DTStatusFlagRcd>().get(statusMap);

  // Get the Phi 2D segment
  int nPhiHits = -1;
  bool segmentNoisyPhi = false;
  if (segment.hasPhi()) {
    const DTChamberRecSegment2D* phiSeg = segment.phiSegment();  // phiSeg lives in the chamber RF
    std::vector<DTRecHit1D> phiRecHits = phiSeg->specificRecHits();
    nPhiHits = phiRecHits.size();
    if (checkNoisyChannels_)
      segmentNoisyPhi = checkNoisySegment(statusMap, phiRecHits);
    //} else result = false;
  }
  // Get the Theta 2D segment
  int nZHits = -1;
  bool segmentNoisyZ = false;
  if (segment.hasZed()) {
    const DTSLRecSegment2D* zSeg = segment.zSegment();  // zSeg lives in the SL RF
    std::vector<DTRecHit1D> zRecHits = zSeg->specificRecHits();
    nZHits = zRecHits.size();
    if (checkNoisyChannels_)
      segmentNoisyZ = checkNoisySegment(statusMap, zRecHits);
  }

  // Segment selection
  // Discard segment if it has a noisy cell
  if (segmentNoisyPhi || segmentNoisyZ)
    result = false;

  // 2D-segment number of hits
  if (segment.hasPhi() && nPhiHits < minHitsPhi_)
    result = false;

  if (segment.hasZed() && nZHits < minHitsZ_)
    result = false;

  // Segment chi2
  double chiSquare = segment.chi2() / segment.degreesOfFreedom();
  if (chiSquare > maxChi2_)
    result = false;

  // Segment angle
  LocalVector segment4DLocalDir = segment.localDirection();
  double angleZ = fabs(atan(segment4DLocalDir.y() / segment4DLocalDir.z()) * 180. / Geom::pi());
  if (angleZ > maxAngleZ_)
    result = false;

  double anglePhi = fabs(atan(segment4DLocalDir.x() / segment4DLocalDir.z()) * 180. / Geom::pi());
  if (anglePhi > maxAnglePhi_)
    result = false;

  return result;
}

bool DTSegmentSelector::checkNoisySegment(edm::ESHandle<DTStatusFlag> const& statusMap,
                                          std::vector<DTRecHit1D> const& dtHits) {
  bool segmentNoisy = false;

  std::vector<DTRecHit1D>::const_iterator dtHit = dtHits.begin();
  std::vector<DTRecHit1D>::const_iterator dtHits_end = dtHits.end();
  for (; dtHit != dtHits_end; ++dtHit) {
    DTWireId wireId = dtHit->wireId();
    // Check for noisy channels to skip them
    bool isNoisy = false, isFEMasked = false, isTDCMasked = false, isTrigMask = false, isDead = false, isNohv = false;
    statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
    if (isNoisy) {
      LogTrace("Calibration") << "Wire: " << wireId << " is noisy, skipping!";
      segmentNoisy = true;
      break;
    }
  }
  return segmentNoisy;
}
