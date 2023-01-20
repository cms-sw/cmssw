#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

using namespace reco;

Muon::Muon(Charge q, const LorentzVector& p4, const Point& vtx) : RecoCandidate(q, p4, vtx, -13 * q) {
  energyValid_ = false;
  matchesValid_ = false;
  isolationValid_ = false;
  pfIsolationValid_ = false;
  qualityValid_ = false;
  caloCompatibility_ = -9999.;
  type_ = 0;
  bestTunePTrackType_ = reco::Muon::None;
  bestTrackType_ = reco::Muon::None;
  selectors_ = 0;
}

Muon::Muon() {
  energyValid_ = false;
  matchesValid_ = false;
  isolationValid_ = false;
  pfIsolationValid_ = false;
  qualityValid_ = false;
  caloCompatibility_ = -9999.;
  type_ = 0;
  bestTrackType_ = reco::Muon::None;
  bestTunePTrackType_ = reco::Muon::None;
  selectors_ = 0;
}

bool Muon::overlap(const Candidate& c) const {
  const RecoCandidate* o = dynamic_cast<const RecoCandidate*>(&c);
  return (o != nullptr && (checkOverlap(track(), o->track()) || checkOverlap(standAloneMuon(), o->standAloneMuon()) ||
                           checkOverlap(combinedMuon(), o->combinedMuon()) ||
                           checkOverlap(standAloneMuon(), o->track()) || checkOverlap(combinedMuon(), o->track())));
}

Muon* Muon::clone() const { return new Muon(*this); }

int Muon::numberOfChambersCSCorDT() const {
  int total = 0;
  int nAll = numberOfChambers();
  for (int iC = 0; iC < nAll; ++iC) {
    if (matches()[iC].detector() == MuonSubdetId::CSC || matches()[iC].detector() == MuonSubdetId::DT)
      total++;
  }

  return total;
}

int Muon::numberOfMatches(ArbitrationType type) const {
  int matches(0);
  for (auto& chamberMatch : muMatches_) {
    if (type == RPCHitAndTrackArbitration) {
      if (chamberMatch.rpcMatches.empty())
        continue;
      matches += chamberMatch.rpcMatches.size();
      continue;
    }
    if (type == ME0SegmentAndTrackArbitration) {
      if (chamberMatch.me0Matches.empty())
        continue;
      matches += chamberMatch.me0Matches.size();
      continue;
    }
    if (type == GEMSegmentAndTrackArbitration) {
      if (chamberMatch.gemMatches.empty())
        continue;
      matches += chamberMatch.gemMatches.size();
      continue;
    }

    if (type == GEMHitAndTrackArbitration) {
      if (chamberMatch.gemHitMatches.empty())
        continue;
      matches += chamberMatch.gemHitMatches.size();
      continue;
    }

    if (chamberMatch.segmentMatches.empty())
      continue;
    if (type == NoArbitration) {
      matches++;
      continue;
    }

    for (auto& segmentMatch : chamberMatch.segmentMatches) {
      if (type == SegmentArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInChamberByDR)) {
          matches++;
          break;
        }
      if (type == SegmentAndTrackArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInChamberByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
          matches++;
          break;
        }
      if (type == SegmentAndTrackArbitrationCleaned)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInChamberByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
          matches++;
          break;
        }
    }
  }

  return matches;
}

int Muon::numberOfMatchedStations(ArbitrationType type) const {
  int stations(0);

  unsigned int theStationMask = stationMask(type);
  // eight stations, eight bits
  for (int it = 0; it < 8; ++it)
    if (theStationMask & 1 << it)
      ++stations;

  return stations;
}

unsigned int Muon::expectedNnumberOfMatchedStations(float minDistanceFromEdge) const {
  unsigned int stationMask = 0;
  minDistanceFromEdge = std::abs(minDistanceFromEdge);
  for (auto& chamberMatch : muMatches_) {
    if (chamberMatch.detector() != MuonSubdetId::DT && chamberMatch.detector() != MuonSubdetId::CSC)
      continue;
    float edgeX = chamberMatch.edgeX;
    float edgeY = chamberMatch.edgeY;
    // check we if the trajectory is well within the acceptance
    if (edgeX < 0 && -edgeX > minDistanceFromEdge && edgeY < 0 && -edgeY > minDistanceFromEdge)
      stationMask |= 1 << ((chamberMatch.station() - 1) + 4 * (chamberMatch.detector() - 1));
  }
  unsigned int n = 0;
  for (unsigned int i = 0; i < 8; ++i)
    if (stationMask & (1 << i))
      n++;
  return n;
}

unsigned int Muon::stationMask(ArbitrationType type) const {
  unsigned int totMask(0);
  unsigned int curMask(0);

  for (auto& chamberMatch : muMatches_) {
    if (type == RPCHitAndTrackArbitration) {
      if (chamberMatch.rpcMatches.empty())
        continue;
      RPCDetId rollId = chamberMatch.id.rawId();
      int rpcIndex = rollId.region() == 0 ? 1 : 2;
      curMask = 1 << ((chamberMatch.station() - 1) + 4 * (rpcIndex - 1));
      // do not double count
      if (!(totMask & curMask))
        totMask += curMask;
      continue;
    }

    if (chamberMatch.segmentMatches.empty())
      continue;
    if (type == NoArbitration) {
      curMask = 1 << ((chamberMatch.station() - 1) + 4 * (chamberMatch.detector() - 1));
      // do not double count
      if (!(totMask & curMask))
        totMask += curMask;
      continue;
    }

    for (auto& segmentMatch : chamberMatch.segmentMatches) {
      if (type == SegmentArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR)) {
          curMask = 1 << ((chamberMatch.station() - 1) + 4 * (chamberMatch.detector() - 1));
          // do not double count
          if (!(totMask & curMask))
            totMask += curMask;
          break;
        }
      if (type == SegmentAndTrackArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
          curMask = 1 << ((chamberMatch.station() - 1) + 4 * (chamberMatch.detector() - 1));
          // do not double count
          if (!(totMask & curMask))
            totMask += curMask;
          break;
        }
      if (type == SegmentAndTrackArbitrationCleaned)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
          curMask = 1 << ((chamberMatch.station() - 1) + 4 * (chamberMatch.detector() - 1));
          // do not double count
          if (!(totMask & curMask))
            totMask += curMask;
          break;
        }
    }
  }

  return totMask;
}

int Muon::numberOfMatchedRPCLayers(ArbitrationType type) const {
  int layers(0);

  unsigned int theRPCLayerMask = RPClayerMask(type);
  // maximum ten layers because of 6 layers in barrel and 3 (4) layers in each endcap before (after) upscope
  for (int it = 0; it < 10; ++it)
    if (theRPCLayerMask & 1 << it)
      ++layers;

  return layers;
}

unsigned int Muon::RPClayerMask(ArbitrationType type) const {
  unsigned int totMask(0);
  unsigned int curMask(0);
  for (auto& chamberMatch : muMatches_) {
    if (chamberMatch.rpcMatches.empty())
      continue;

    RPCDetId rollId = chamberMatch.id.rawId();
    const int region = rollId.region();
    const int stationIndex = chamberMatch.station();
    int rpcLayer = stationIndex;
    if (region == 0) {
      const int layer = rollId.layer();
      rpcLayer = stationIndex - 1 + stationIndex * layer;
      if ((stationIndex == 2 && layer == 2) || (stationIndex == 4 && layer == 1))
        rpcLayer -= 1;
    } else
      rpcLayer += 6;

    curMask = 1 << (rpcLayer - 1);
    // do not double count
    if (!(totMask & curMask))
      totMask += curMask;
  }

  return totMask;
}

unsigned int Muon::stationGapMaskDistance(float distanceCut) const {
  unsigned int totMask(0);
  distanceCut = std::abs(distanceCut);
  for (auto& chamberMatch : muMatches_) {
    unsigned int curMask(0);
    const int detectorIndex = chamberMatch.detector();
    if (detectorIndex < 1 || detectorIndex >= 4)
      continue;
    const int stationIndex = chamberMatch.station();
    if (stationIndex < 1 || stationIndex >= 5)
      continue;

    float edgeX = chamberMatch.edgeX;
    float edgeY = chamberMatch.edgeY;
    if (edgeX < 0 && -edgeX > distanceCut && edgeY < 0 &&
        -edgeY > distanceCut)  // inside the chamber so negates all gaps for this station
      continue;

    if ((std::abs(edgeX) < distanceCut && edgeY < distanceCut) ||
        (std::abs(edgeY) < distanceCut && edgeX < distanceCut))  // inside gap
      curMask = 1 << ((stationIndex - 1) + 4 * (detectorIndex - 1));
    totMask += curMask;  // add to total mask
  }

  return totMask;
}

unsigned int Muon::stationGapMaskPull(float sigmaCut) const {
  sigmaCut = std::abs(sigmaCut);
  unsigned int totMask(0);
  for (auto& chamberMatch : muMatches_) {
    unsigned int curMask(0);
    const int detectorIndex = chamberMatch.detector();
    if (detectorIndex < 1 || detectorIndex >= 4)
      continue;
    const int stationIndex = chamberMatch.station();
    if (stationIndex < 1 || stationIndex >= 5)
      continue;

    float edgeX = chamberMatch.edgeX;
    float edgeY = chamberMatch.edgeY;
    float xErr = chamberMatch.xErr + 0.000001;  // protect against division by zero later
    float yErr = chamberMatch.yErr + 0.000001;  // protect against division by zero later
    if (edgeX < 0 && std::abs(edgeX / xErr) > sigmaCut && edgeY < 0 &&
        std::abs(edgeY / yErr) > sigmaCut)  // inside the chamber so negates all gaps for this station
      continue;

    if ((std::abs(edgeX / xErr) < sigmaCut && edgeY < sigmaCut * yErr) ||
        (std::abs(edgeY / yErr) < sigmaCut && edgeX < sigmaCut * xErr))  // inside gap
      curMask = 1 << ((stationIndex - 1) + 4 * (detectorIndex - 1));
    totMask += curMask;  // add to total mask
  }

  return totMask;
}

int Muon::nDigisInStation(int station, int muonSubdetId) const {
  int nDigis(0);
  std::map<int, int> me11DigisPerCh;

  if (muonSubdetId != MuonSubdetId::CSC && muonSubdetId != MuonSubdetId::DT)
    return 0;

  for (auto& match : muMatches_) {
    if (match.detector() != muonSubdetId || match.station() != station)
      continue;

    int nDigisInCh = match.nDigisInRange;

    if (muonSubdetId == MuonSubdetId::CSC && station == 1) {
      CSCDetId id(match.id.rawId());

      int chamber = id.chamber();
      int ring = id.ring();

      if (ring == 1 || ring == 4)  // merge ME1/1a and ME1/1b digis
      {
        if (me11DigisPerCh.find(chamber) == me11DigisPerCh.end())
          me11DigisPerCh[chamber] = 0;

        me11DigisPerCh[chamber] += nDigisInCh;

        continue;
      }
    }

    if (nDigisInCh > nDigis)
      nDigis = nDigisInCh;
  }

  for (const auto& me11DigisInCh : me11DigisPerCh) {
    int nMe11DigisInCh = me11DigisInCh.second;
    if (nMe11DigisInCh > nDigis)
      nDigis = nMe11DigisInCh;
  }

  return nDigis;
}

bool Muon::hasShowerInStation(int station, int muonSubdetId, int nDtDigisCut, int nCscDigisCut) const {
  if (muonSubdetId != MuonSubdetId::DT && muonSubdetId != MuonSubdetId::CSC)
    return false;
  auto nDigisCut = muonSubdetId == MuonSubdetId::DT ? nDtDigisCut : nCscDigisCut;

  return nDigisInStation(station, muonSubdetId) >= nDigisCut;
}

int Muon::numberOfShowers(int nDtDigisCut, int nCscDigisCut) const {
  int nShowers = 0;
  for (int station = 1; station < 5; ++station) {
    if (hasShowerInStation(station, MuonSubdetId::DT, nDtDigisCut, nCscDigisCut))
      nShowers++;
    if (hasShowerInStation(station, MuonSubdetId::CSC, nDtDigisCut, nCscDigisCut))
      nShowers++;
  }

  return nShowers;
}

int Muon::numberOfSegments(int station, int muonSubdetId, ArbitrationType type) const {
  int segments(0);
  for (auto& chamberMatch : muMatches_) {
    if (chamberMatch.segmentMatches.empty())
      continue;
    if (chamberMatch.detector() != muonSubdetId || chamberMatch.station() != station)
      continue;

    if (type == NoArbitration) {
      segments += chamberMatch.segmentMatches.size();
      continue;
    }

    for (auto& segmentMatch : chamberMatch.segmentMatches) {
      if (type == SegmentArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR)) {
          segments++;
          break;
        }
      if (type == SegmentAndTrackArbitration)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
          segments++;
          break;
        }
      if (type == SegmentAndTrackArbitrationCleaned)
        if (segmentMatch.isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
            segmentMatch.isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
          segments++;
          break;
        }
    }
  }

  return segments;
}

const std::vector<const MuonChamberMatch*> Muon::chambers(int station, int muonSubdetId) const {
  std::vector<const MuonChamberMatch*> chambers;
  for (std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
       chamberMatch != muMatches_.end();
       chamberMatch++)
    if (chamberMatch->detector() == muonSubdetId && chamberMatch->station() == station)
      chambers.push_back(&(*chamberMatch));
  return chambers;
}

std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> Muon::pair(
    const std::vector<const MuonChamberMatch*>& chambers, ArbitrationType type) const {
  MuonChamberMatch* m = nullptr;
  MuonSegmentMatch* s = nullptr;
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair(m, s);

  if (chambers.empty())
    return chamberSegmentPair;
  for (std::vector<const MuonChamberMatch*>::const_iterator chamberMatch = chambers.begin();
       chamberMatch != chambers.end();
       chamberMatch++) {
    if ((*chamberMatch)->segmentMatches.empty())
      continue;
    if (type == NoArbitration)
      return std::make_pair(*chamberMatch, &((*chamberMatch)->segmentMatches.front()));

    for (std::vector<MuonSegmentMatch>::const_iterator segmentMatch = (*chamberMatch)->segmentMatches.begin();
         segmentMatch != (*chamberMatch)->segmentMatches.end();
         segmentMatch++) {
      if (type == SegmentArbitration)
        if (segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR))
          return std::make_pair(*chamberMatch, &(*segmentMatch));
      if (type == SegmentAndTrackArbitration)
        if (segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR))
          return std::make_pair(*chamberMatch, &(*segmentMatch));
      if (type == SegmentAndTrackArbitrationCleaned)
        if (segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
            segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
            segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByCleaning))
          return std::make_pair(*chamberMatch, &(*segmentMatch));
    }
  }

  return chamberSegmentPair;
}

float Muon::dX(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.first->x - chamberSegmentPair.second->x;
}

float Muon::dY(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.first->y - chamberSegmentPair.second->y;
}

float Muon::dDxDz(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.first->dXdZ - chamberSegmentPair.second->dXdZ;
}

float Muon::dDyDz(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.first->dYdZ - chamberSegmentPair.second->dYdZ;
}

float Muon::pullX(int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  if (includeSegmentError)
    return (chamberSegmentPair.first->x - chamberSegmentPair.second->x) /
           sqrt(std::pow(chamberSegmentPair.first->xErr, 2) + std::pow(chamberSegmentPair.second->xErr, 2));
  return (chamberSegmentPair.first->x - chamberSegmentPair.second->x) / chamberSegmentPair.first->xErr;
}

float Muon::pullY(int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  if (includeSegmentError)
    return (chamberSegmentPair.first->y - chamberSegmentPair.second->y) /
           sqrt(std::pow(chamberSegmentPair.first->yErr, 2) + std::pow(chamberSegmentPair.second->yErr, 2));
  return (chamberSegmentPair.first->y - chamberSegmentPair.second->y) / chamberSegmentPair.first->yErr;
}

float Muon::pullDxDz(int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  if (includeSegmentError)
    return (chamberSegmentPair.first->dXdZ - chamberSegmentPair.second->dXdZ) /
           sqrt(std::pow(chamberSegmentPair.first->dXdZErr, 2) + std::pow(chamberSegmentPair.second->dXdZErr, 2));
  return (chamberSegmentPair.first->dXdZ - chamberSegmentPair.second->dXdZ) / chamberSegmentPair.first->dXdZErr;
}

float Muon::pullDyDz(int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  if (includeSegmentError)
    return (chamberSegmentPair.first->dYdZ - chamberSegmentPair.second->dYdZ) /
           sqrt(std::pow(chamberSegmentPair.first->dYdZErr, 2) + std::pow(chamberSegmentPair.second->dYdZErr, 2));
  return (chamberSegmentPair.first->dYdZ - chamberSegmentPair.second->dYdZ) / chamberSegmentPair.first->dYdZErr;
}

float Muon::segmentX(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.second->x;
}

float Muon::segmentY(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.second->y;
}

float Muon::segmentDxDz(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.second->dXdZ;
}

float Muon::segmentDyDz(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.second->dYdZ;
}

float Muon::segmentXErr(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.second->xErr;
}

float Muon::segmentYErr(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.second->yErr;
}

float Muon::segmentDxDzErr(int station, int muonSubdetId, ArbitrationType type) const {
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasPhi())
    return 999999;
  return chamberSegmentPair.second->dXdZErr;
}

float Muon::segmentDyDzErr(int station, int muonSubdetId, ArbitrationType type) const {
  if (station == 4 && muonSubdetId == MuonSubdetId::DT)
    return 999999;  // no y information
  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair =
      pair(chambers(station, muonSubdetId), type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr)
    return 999999;
  if (!chamberSegmentPair.second->hasZed())
    return 999999;
  return chamberSegmentPair.second->dYdZErr;
}

float Muon::trackEdgeX(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->edgeX;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->edgeX;
}

float Muon::trackEdgeY(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->edgeY;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->edgeY;
}

float Muon::trackX(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->x;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->x;
}

float Muon::trackY(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->y;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->y;
}

float Muon::trackDxDz(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->dXdZ;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->dXdZ;
}

float Muon::trackDyDz(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->dYdZ;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->dYdZ;
}

float Muon::trackXErr(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->xErr;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->xErr;
}

float Muon::trackYErr(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->yErr;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->yErr;
}

float Muon::trackDxDzErr(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->dXdZErr;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->dXdZErr;
}

float Muon::trackDyDzErr(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->dYdZErr;
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->dYdZErr;
}

float Muon::trackDist(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist)
        dist = currDist;
    }
    return dist;
  } else
    return chamberSegmentPair.first->dist();
}

float Muon::trackDistErr(int station, int muonSubdetId, ArbitrationType type) const {
  const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
  if (muonChambers.empty())
    return 999999;

  std::pair<const MuonChamberMatch*, const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers, type);
  if (chamberSegmentPair.first == nullptr || chamberSegmentPair.second == nullptr) {
    float dist = 999999;
    float supVar = 999999;
    for (const MuonChamberMatch* muonChamber : muonChambers) {
      float currDist = muonChamber->dist();
      if (currDist < dist) {
        dist = currDist;
        supVar = muonChamber->distErr();
      }
    }
    return supVar;
  } else
    return chamberSegmentPair.first->distErr();
}

void Muon::setIsolation(const MuonIsolation& isoR03, const MuonIsolation& isoR05) {
  isolationR03_ = isoR03;
  isolationR05_ = isoR05;
  isolationValid_ = true;
}

void Muon::setPFIsolation(const std::string& label, const MuonPFIsolation& deposit) {
  if (label == "pfIsolationR03")
    pfIsolationR03_ = deposit;

  if (label == "pfIsolationR04")
    pfIsolationR04_ = deposit;

  if (label == "pfIsoMeanDRProfileR03")
    pfIsoMeanDRR03_ = deposit;

  if (label == "pfIsoMeanDRProfileR04")
    pfIsoMeanDRR04_ = deposit;

  if (label == "pfIsoSumDRProfileR03")
    pfIsoSumDRR03_ = deposit;

  if (label == "pfIsoSumDRProfileR04")
    pfIsoSumDRR04_ = deposit;

  pfIsolationValid_ = true;
}

void Muon::setPFP4(const reco::Candidate::LorentzVector& p4) {
  pfP4_ = p4;
  type_ = type_ | PFMuon;
}

void Muon::setOuterTrack(const TrackRef& t) { outerTrack_ = t; }
void Muon::setInnerTrack(const TrackRef& t) { innerTrack_ = t; }
void Muon::setTrack(const TrackRef& t) { setInnerTrack(t); }
void Muon::setStandAlone(const TrackRef& t) { setOuterTrack(t); }
void Muon::setGlobalTrack(const TrackRef& t) { globalTrack_ = t; }
void Muon::setCombined(const TrackRef& t) { setGlobalTrack(t); }

bool Muon::isAValidMuonTrack(const MuonTrackType& type) const { return muonTrack(type).isNonnull(); }

TrackRef Muon::muonTrack(const MuonTrackType& type) const {
  switch (type) {
    case InnerTrack:
      return innerTrack();
    case OuterTrack:
      return standAloneMuon();
    case CombinedTrack:
      return globalTrack();
    case TPFMS:
      return tpfmsTrack();
    case Picky:
      return pickyTrack();
    case DYT:
      return dytTrack();
    default:
      return muonTrackFromMap(type);
  }
}

void Muon::setMuonTrack(const MuonTrackType& type, const TrackRef& t) {
  switch (type) {
    case InnerTrack:
      setInnerTrack(t);
      break;
    case OuterTrack:
      setStandAlone(t);
      break;
    case CombinedTrack:
      setGlobalTrack(t);
      break;
    default:
      refittedTrackMap_[type] = t;
      break;
  }
}
