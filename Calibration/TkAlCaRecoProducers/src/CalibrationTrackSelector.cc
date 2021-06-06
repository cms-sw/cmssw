#include "Calibration/TkAlCaRecoProducers/interface/CalibrationTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

// constructor ----------------------------------------------------------------

CalibrationTrackSelector::CalibrationTrackSelector(const edm::ParameterSet &cfg)
    : applyBasicCuts_(cfg.getParameter<bool>("applyBasicCuts")),
      applyNHighestPt_(cfg.getParameter<bool>("applyNHighestPt")),
      applyMultiplicityFilter_(cfg.getParameter<bool>("applyMultiplicityFilter")),
      seedOnlyFromAbove_(cfg.getParameter<int>("seedOnlyFrom")),
      applyIsolation_(cfg.getParameter<bool>("applyIsolationCut")),
      chargeCheck_(cfg.getParameter<bool>("applyChargeCheck")),
      nHighestPt_(cfg.getParameter<int>("nHighestPt")),
      minMultiplicity_(cfg.getParameter<int>("minMultiplicity")),
      maxMultiplicity_(cfg.getParameter<int>("maxMultiplicity")),
      multiplicityOnInput_(cfg.getParameter<bool>("multiplicityOnInput")),
      ptMin_(cfg.getParameter<double>("ptMin")),
      ptMax_(cfg.getParameter<double>("ptMax")),
      etaMin_(cfg.getParameter<double>("etaMin")),
      etaMax_(cfg.getParameter<double>("etaMax")),
      phiMin_(cfg.getParameter<double>("phiMin")),
      phiMax_(cfg.getParameter<double>("phiMax")),
      nHitMin_(cfg.getParameter<double>("nHitMin")),
      nHitMax_(cfg.getParameter<double>("nHitMax")),
      chi2nMax_(cfg.getParameter<double>("chi2nMax")),
      minHitChargeStrip_(cfg.getParameter<double>("minHitChargeStrip")),
      minHitIsolation_(cfg.getParameter<double>("minHitIsolation")),
      rphirecHitsTag_(cfg.getParameter<edm::InputTag>("rphirecHits")),
      matchedrecHitsTag_(cfg.getParameter<edm::InputTag>("matchedrecHits")),
      nHitMin2D_(cfg.getParameter<unsigned int>("nHitMin2D")),
      // Ugly to use the same getParameter 6 times, but this allows const cut
      // variables...
      minHitsinTIB_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inTIB")),
      minHitsinTOB_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inTOB")),
      minHitsinTID_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inTID")),
      minHitsinTEC_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inTEC")),
      minHitsinBPIX_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inBPIX")),
      minHitsinFPIX_(cfg.getParameter<edm::ParameterSet>("minHitsPerSubDet").getParameter<int>("inFPIX")) {
  if (applyBasicCuts_)
    edm::LogInfo("CalibrationTrackSelector")
        << "applying basic track cuts ..."
        << "\nptmin,ptmax:     " << ptMin_ << "," << ptMax_ << "\netamin,etamax:   " << etaMin_ << "," << etaMax_
        << "\nphimin,phimax:   " << phiMin_ << "," << phiMax_ << "\nnhitmin,nhitmax: " << nHitMin_ << "," << nHitMax_
        << "\nnhitmin2D:       " << nHitMin2D_ << "\nchi2nmax:        " << chi2nMax_;

  if (applyNHighestPt_)
    edm::LogInfo("CalibrationTrackSelector") << "filter N tracks with highest Pt N=" << nHighestPt_;

  if (applyMultiplicityFilter_)
    edm::LogInfo("CalibrationTrackSelector")
        << "apply multiplicity filter N>= " << minMultiplicity_ << "and N<= " << maxMultiplicity_ << " on "
        << (multiplicityOnInput_ ? "input" : "output");

  if (applyIsolation_)
    edm::LogInfo("CalibrationTrackSelector")
        << "only retain tracks isolated at least by " << minHitIsolation_ << " cm from other rechits";

  if (chargeCheck_)
    edm::LogInfo("CalibrationTrackSelector")
        << "only retain hits with at least " << minHitChargeStrip_ << " ADC counts of total cluster charge";

  edm::LogInfo("CalibrationTrackSelector")
      << "Minimum number of hits in TIB/TID/TOB/TEC/BPIX/FPIX = " << minHitsinTIB_ << "/" << minHitsinTID_ << "/"
      << minHitsinTOB_ << "/" << minHitsinTEC_ << "/" << minHitsinBPIX_ << "/" << minHitsinFPIX_;
}

// destructor -----------------------------------------------------------------

CalibrationTrackSelector::~CalibrationTrackSelector() {}

// do selection ---------------------------------------------------------------

CalibrationTrackSelector::Tracks CalibrationTrackSelector::select(const Tracks &tracks, const edm::Event &evt) const {
  if (applyMultiplicityFilter_ && multiplicityOnInput_ &&
      (tracks.size() < static_cast<unsigned int>(minMultiplicity_) ||
       tracks.size() > static_cast<unsigned int>(maxMultiplicity_))) {
    return Tracks();  // empty collection
  }

  Tracks result = tracks;
  // apply basic track cuts (if selected)
  if (applyBasicCuts_)
    result = this->basicCuts(result, evt);

  // filter N tracks with highest Pt (if selected)
  if (applyNHighestPt_)
    result = this->theNHighestPtTracks(result);

  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter_ && !multiplicityOnInput_) {
    if (result.size() < static_cast<unsigned int>(minMultiplicity_) ||
        result.size() > static_cast<unsigned int>(maxMultiplicity_)) {
      result.clear();
    }
  }

  // edm::LogDebug("CalibrationTrackSelector") << "tracks all,kept: " <<
  // tracks.size() << "," << result.size();

  return result;
}

// make basic cuts ------------------------------------------------------------

CalibrationTrackSelector::Tracks CalibrationTrackSelector::basicCuts(const Tracks &tracks,
                                                                     const edm::Event &evt) const {
  Tracks result;

  for (Tracks::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
    const reco::Track *trackp = *it;
    float pt = trackp->pt();
    float eta = trackp->eta();
    float phi = trackp->phi();
    int nhit = trackp->numberOfValidHits();
    float chi2n = trackp->normalizedChi2();

    // edm::LogDebug("CalibrationTrackSelector") << " pt,eta,phi,nhit: "
    //  <<pt<<","<<eta<<","<<phi<<","<<nhit;

    if (pt > ptMin_ && pt < ptMax_ && eta > etaMin_ && eta < etaMax_ && phi > phiMin_ && phi < phiMax_ &&
        nhit >= nHitMin_ && nhit <= nHitMax_ && chi2n < chi2nMax_) {
      if (this->detailedHitsCheck(trackp, evt))
        result.push_back(trackp);
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

bool CalibrationTrackSelector::detailedHitsCheck(const reco::Track *trackp, const edm::Event &evt) const {
  // checking hit requirements beyond simple number of valid hits

  if (minHitsinTIB_ || minHitsinTOB_ || minHitsinTID_ || minHitsinTEC_ || minHitsinFPIX_ || minHitsinBPIX_ ||
      nHitMin2D_ || chargeCheck_ || applyIsolation_) {  // any detailed hit cut is active, so have to check

    int nhitinTIB = 0, nhitinTOB = 0, nhitinTID = 0;
    int nhitinTEC = 0, nhitinBPIX = 0, nhitinFPIX = 0;
    unsigned int nHit2D = 0;
    unsigned int thishit = 0;

    for (auto const &hit : trackp->recHits()) {
      thishit++;
      int type = hit->geographicalId().subdetId();

      // *** thishit == 1 means last hit in CTF ***
      // (FIXME: assumption might change or not be valid for all tracking
      // algorthms)
      // ==> for cosmics
      // seedOnlyFrom = 1 is TIB-TOB-TEC tracks only
      // seedOnlyFrom = 2 is TOB-TEC tracks only
      if (seedOnlyFromAbove_ == 1 && thishit == 1 &&
          (type == int(StripSubdetector::TOB) || type == int(StripSubdetector::TEC)))
        return false;

      if (seedOnlyFromAbove_ == 2 && thishit == 1 && type == int(StripSubdetector::TIB))
        return false;

      if (!hit->isValid())
        continue;  // only real hits count as in trackp->numberOfValidHits()
      const DetId detId(hit->geographicalId());
      if (detId.det() != DetId::Tracker) {
        edm::LogError("DetectorMismatch")
            << "@SUB=CalibrationTrackSelector::detailedHitsCheck"
            << "DetId.det() != DetId::Tracker (=" << DetId::Tracker << "), but " << detId.det() << ".";
      }
      if (chargeCheck_ && !(this->isOkCharge(hit)))
        return false;
      if (applyIsolation_ && (!this->isIsolated(hit, evt)))
        return false;
      if (StripSubdetector::TIB == detId.subdetId())
        ++nhitinTIB;
      else if (StripSubdetector::TOB == detId.subdetId())
        ++nhitinTOB;
      else if (StripSubdetector::TID == detId.subdetId())
        ++nhitinTID;
      else if (StripSubdetector::TEC == detId.subdetId())
        ++nhitinTEC;
      else if (kBPIX == detId.subdetId())
        ++nhitinBPIX;
      else if (kFPIX == detId.subdetId())
        ++nhitinFPIX;
      // Do not call isHit2D(..) if already enough 2D hits for performance
      // reason:
      if (nHit2D < nHitMin2D_ && this->isHit2D(*hit))
        ++nHit2D;
    }  // end loop on hits
    return (nhitinTIB >= minHitsinTIB_ && nhitinTOB >= minHitsinTOB_ && nhitinTID >= minHitsinTID_ &&
            nhitinTEC >= minHitsinTEC_ && nhitinBPIX >= minHitsinBPIX_ && nhitinFPIX >= minHitsinFPIX_ &&
            nHit2D >= nHitMin2D_);
  } else {  // no cuts set, so we are just fine and can avoid loop on hits
    return true;
  }
}

//-----------------------------------------------------------------------------

bool CalibrationTrackSelector::isHit2D(const TrackingRecHit &hit) const {
  if (hit.dimension() < 2) {
    return false;  // some (muon...) stuff really has RecHit1D
  } else {
    const DetId detId(hit.geographicalId());
    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
        return true;  // pixel is always 2D
      } else {        // should be SiStrip now
        if (dynamic_cast<const SiStripRecHit2D *>(&hit))
          return false;  // normal hit
        else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))
          return true;  // matched is 2D
        else if (dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit))
          return false;  // crazy hit...
        else {
          edm::LogError("UnknownType") << "@SUB=CalibrationTrackSelector::isHit2D"
                                       << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
                                       << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
          return false;
        }
      }
    } else {  // not tracker??
      edm::LogWarning("DetectorMismatch") << "@SUB=CalibrationTrackSelector::isHit2D"
                                          << "Hit not in tracker with 'official' dimension >=2.";
      return true;  // dimension() >= 2 so accept that...
    }
  }
  // never reached...
}

//-----------------------------------------------------------------------------

bool CalibrationTrackSelector::isOkCharge(const TrackingRecHit *therechit) const {
  float charge1 = 0;
  float charge2 = 0;
  const SiStripMatchedRecHit2D *matchedhit = dynamic_cast<const SiStripMatchedRecHit2D *>(therechit);
  const SiStripRecHit2D *hit = dynamic_cast<const SiStripRecHit2D *>(therechit);
  const ProjectedSiStripRecHit2D *unmatchedhit = dynamic_cast<const ProjectedSiStripRecHit2D *>(therechit);

  if (matchedhit) {
    const SiStripCluster &monocluster = matchedhit->monoCluster();
    const std::vector<uint16_t> amplitudesmono(monocluster.amplitudes().begin(), monocluster.amplitudes().end());
    for (size_t ia = 0; ia < amplitudesmono.size(); ++ia) {
      charge1 += amplitudesmono[ia];
    }

    const SiStripCluster &stereocluster = matchedhit->stereoCluster();
    const std::vector<uint16_t> amplitudesstereo(stereocluster.amplitudes().begin(), stereocluster.amplitudes().end());
    for (size_t ia = 0; ia < amplitudesstereo.size(); ++ia) {
      charge2 += amplitudesstereo[ia];
    }
    // std::cout << "charge1 = " << charge1 << "\n";
    // std::cout << "charge2 = " << charge2 << "\n";
    if (charge1 < minHitChargeStrip_ || charge2 < minHitChargeStrip_)
      return false;
  } else if (hit) {
    const SiStripCluster *cluster = &*(hit->cluster());
    const std::vector<uint16_t> amplitudes(cluster->amplitudes().begin(), cluster->amplitudes().end());
    for (size_t ia = 0; ia < amplitudes.size(); ++ia) {
      charge1 += amplitudes[ia];
    }
    // std::cout << "charge1 = " << charge1 << "\n";
    if (charge1 < minHitChargeStrip_)
      return false;
  } else if (unmatchedhit) {
    const SiStripRecHit2D &orighit = unmatchedhit->originalHit();
    const SiStripCluster *origcluster = &*(orighit.cluster());
    const std::vector<uint16_t> amplitudes(origcluster->amplitudes().begin(), origcluster->amplitudes().end());
    for (size_t ia = 0; ia < amplitudes.size(); ++ia) {
      charge1 += amplitudes[ia];
    }
    // std::cout << "charge1 = " << charge1 << "\n";
    if (charge1 < minHitChargeStrip_)
      return false;
  }
  return true;
}

//-----------------------------------------------------------------------------

bool CalibrationTrackSelector::isIsolated(const TrackingRecHit *therechit, const edm::Event &evt) const {
  // edm::ESHandle<TrackerGeometry> tracker;
  edm::Handle<SiStripRecHit2DCollection> rphirecHits;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
  // es.get<TrackerDigiGeometryRecord>().get(tracker);
  evt.getByLabel(rphirecHitsTag_, rphirecHits);
  evt.getByLabel(matchedrecHitsTag_, matchedrecHits);

  SiStripRecHit2DCollection::DataContainer::const_iterator istripSt;
  SiStripMatchedRecHit2DCollection::DataContainer::const_iterator istripStm;
  const SiStripRecHit2DCollection &stripcollSt = *rphirecHits;
  const SiStripMatchedRecHit2DCollection &stripcollStm = *matchedrecHits;

  DetId idet = therechit->geographicalId();

  // FIXME: instead of looping the full hit collection, we should explore the
  // features of SiStripRecHit2DCollection::rangeRphi = rphirecHits.get(idet)
  // and loop only from rangeRphi.first until rangeRphi.second
  for (istripSt = stripcollSt.data().begin(); istripSt != stripcollSt.data().end(); ++istripSt) {
    const SiStripRecHit2D *aHit = &*(istripSt);
    DetId mydet1 = aHit->geographicalId();
    if (idet.rawId() != mydet1.rawId())
      continue;
    float theDistance = (therechit->localPosition() - aHit->localPosition()).mag();
    // std::cout << "theDistance1 = " << theDistance << "\n";
    if (theDistance > 0.001 && theDistance < minHitIsolation_)
      return false;
  }

  // FIXME: see above
  for (istripStm = stripcollStm.data().begin(); istripStm != stripcollStm.data().end(); ++istripStm) {
    const SiStripMatchedRecHit2D *aHit = &*(istripStm);
    DetId mydet2 = aHit->geographicalId();
    if (idet.rawId() != mydet2.rawId())
      continue;
    float theDistance = (therechit->localPosition() - aHit->localPosition()).mag();
    // std::cout << "theDistance1 = " << theDistance << "\n";
    if (theDistance > 0.001 && theDistance < minHitIsolation_)
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------

CalibrationTrackSelector::Tracks CalibrationTrackSelector::theNHighestPtTracks(const Tracks &tracks) const {
  Tracks sortedTracks = tracks;
  Tracks result;

  // sort in pt
  std::sort(sortedTracks.begin(), sortedTracks.end(), ptComparator);

  // copy theTrackMult highest pt tracks to result vector
  int n = 0;
  for (Tracks::const_iterator it = sortedTracks.begin(); it != sortedTracks.end(); ++it) {
    if (n < nHighestPt_) {
      result.push_back(*it);
      n++;
    }
  }

  return result;
}
