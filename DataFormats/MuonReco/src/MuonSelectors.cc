#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/MuonRPCHitMatch.h"

namespace muon {
  SelectionType selectionTypeFromString(const std::string& label) {
    SelectionType value = (SelectionType)-1;
    bool found = false;
    for (int i = 0; selectionTypeStringToEnumMap[i].label && (!found); ++i)
      if (!strcmp(label.c_str(), selectionTypeStringToEnumMap[i].label)) {
        found = true;
        value = selectionTypeStringToEnumMap[i].value;
      }

    // in case of unrecognized selection type
    if (!found)
      throw cms::Exception("MuonSelectorError") << label << " is not a recognized SelectionType";
    return value;
  }

  reco::Muon::Selector selectorFromString(const std::string& label) {
    reco::Muon::Selector value = (reco::Muon::Selector)-1;
    bool found = false;
    for (int i = 0; selectorStringToEnumMap[i].label && (!found); ++i)
      if (!strcmp(label.c_str(), selectorStringToEnumMap[i].label)) {
        found = true;
        value = selectorStringToEnumMap[i].value;
      }

    // in case of unrecognized selection type
    if (!found)
      throw cms::Exception("MuonSelectorError") << label << " is not a recognized reco::Muon::Selector";
    return value;
  }
}  // namespace muon

unsigned int muon::RequiredStationMask(const reco::Muon& muon,
                                       double maxChamberDist,
                                       double maxChamberDistPull,
                                       reco::Muon::ArbitrationType arbitrationType) {
  unsigned int theMask = 0;

  for (int stationIdx = 1; stationIdx < 5; ++stationIdx) {
    for (int detectorIdx = 1; detectorIdx < 3; ++detectorIdx) {
      float dist = muon.trackDist(stationIdx, detectorIdx, arbitrationType);
      if (dist < maxChamberDist &&
          dist < maxChamberDistPull * muon.trackDistErr(stationIdx, detectorIdx, arbitrationType))
        theMask += 1 << ((stationIdx - 1) + 4 * (detectorIdx - 1));
    }
  }
  return theMask;
}

// ------------ method to calculate the calo compatibility for a track with matched muon info  ------------
float muon::caloCompatibility(const reco::Muon& muon) { return muon.caloCompatibility(); }

// ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
float muon::segmentCompatibility(const reco::Muon& muon, reco::Muon::ArbitrationType arbitrationType) {
  bool use_weight_regain_at_chamber_boundary = true;
  bool use_match_dist_penalty = true;

  int nr_of_stations_crossed = 0;
  int nr_of_stations_with_segment = 0;
  std::vector<int> stations_w_track(8);
  std::vector<int> station_has_segmentmatch(8);
  std::vector<int> station_was_crossed(8);
  std::vector<float> stations_w_track_at_boundary(8);
  std::vector<float> station_weight(8);
  int position_in_stations = 0;
  float full_weight = 0.;

  for (int i = 1; i <= 8; ++i) {
    // ********************************************************;
    // *** fill local info for this muon (do some counting) ***;
    // ************** begin ***********************************;
    if (i <= 4) {  // this is the section for the DTs
      float thisTrackDist = muon.trackDist(i, 1, arbitrationType);
      if (thisTrackDist < 999999) {  //current "raw" info that a track is close to a chamber
        ++nr_of_stations_crossed;
        station_was_crossed[i - 1] = 1;
        if (thisTrackDist > -10.)
          stations_w_track_at_boundary[i - 1] = thisTrackDist;
        else
          stations_w_track_at_boundary[i - 1] = 0.;
      }
      //current "raw" info that a segment is matched to the current track
      if (muon.segmentX(i, 1, arbitrationType) < 999999) {
        ++nr_of_stations_with_segment;
        station_has_segmentmatch[i - 1] = 1;
      }
    } else {  // this is the section for the CSCs
      float thisTrackDist = muon.trackDist(i - 4, 2, arbitrationType);
      if (thisTrackDist < 999999) {  //current "raw" info that a track is close to a chamber
        ++nr_of_stations_crossed;
        station_was_crossed[i - 1] = 1;
        if (thisTrackDist > -10.)
          stations_w_track_at_boundary[i - 1] = thisTrackDist;
        else
          stations_w_track_at_boundary[i - 1] = 0.;
      }
      //current "raw" info that a segment is matched to the current track
      if (muon.segmentX(i - 4, 2, arbitrationType) < 999999) {
        ++nr_of_stations_with_segment;
        station_has_segmentmatch[i - 1] = 1;
      }
    }
    // rough estimation of chamber border efficiency (should be parametrized better, this is just a quick guess):
    // TF1 * merf = new TF1("merf","-0.5*(TMath::Erf(x/6.)-1)",-100,100);
    // use above value to "unpunish" missing segment if close to border, i.e. rather than not adding any weight, add
    // the one from the function. Only for dist ~> -10 cm, else full punish!.

    // ********************************************************;
    // *** fill local info for this muon (do some counting) ***;
    // ************** end *************************************;
  }

  // ********************************************************;
  // *** calculate weights for each station *****************;
  // ************** begin ***********************************;
  //    const float slope = 0.5;
  //    const float attenuate_weight_regain = 1.;
  // if attenuate_weight_regain < 1., additional punishment if track is close to boundary and no segment
  const float attenuate_weight_regain = 0.5;

  for (int i = 1; i <= 8; ++i) {  // loop over all possible stations

    // first set all weights if a station has been crossed
    // later penalize if a station did not have a matching segment

    //old logic      if(station_has_segmentmatch[i-1] > 0 ) { // the track has an associated segment at the current station
    if (station_was_crossed[i - 1] > 0) {  // the track crossed this chamber (or was nearby)
      // - Apply a weight depending on the "depth" of the muon passage.
      // - The station_weight is later reduced for stations with badly matched segments.
      // - Even if there is no segment but the track passes close to a chamber boundary, the
      //   weight is set non zero and can go up to 0.5 of the full weight if the track is quite
      //   far from any station.
      ++position_in_stations;

      switch (nr_of_stations_crossed) {  // define different weights depending on how many stations were crossed
        case 1:
          station_weight[i - 1] = 1.f;
          break;
        case 2:
          if (position_in_stations == 1)
            station_weight[i - 1] = 0.33f;
          else
            station_weight[i - 1] = 0.67f;
          break;
        case 3:
          if (position_in_stations == 1)
            station_weight[i - 1] = 0.23f;
          else if (position_in_stations == 2)
            station_weight[i - 1] = 0.33f;
          else
            station_weight[i - 1] = 0.44f;
          break;
        case 4:
          if (position_in_stations == 1)
            station_weight[i - 1] = 0.10f;
          else if (position_in_stations == 2)
            station_weight[i - 1] = 0.20f;
          else if (position_in_stations == 3)
            station_weight[i - 1] = 0.30f;
          else
            station_weight[i - 1] = 0.40f;
          break;

        default:
          // 	LogTrace("MuonIdentification")<<"            // Message: A muon candidate track has more than 4 stations with matching segments.";
          // 	LogTrace("MuonIdentification")<<"            // Did not expect this - please let me know: ibloch@fnal.gov";
          // for all other cases
          station_weight[i - 1] = 1.f / nr_of_stations_crossed;
      }

      if (use_weight_regain_at_chamber_boundary) {  // reconstitute some weight if there is no match but the segment is close to a boundary:
        if (station_has_segmentmatch[i - 1] <= 0 && stations_w_track_at_boundary[i - 1] != 0.) {
          // if segment is not present but track in inefficient region, do not count as "missing match" but add some reduced weight.
          // original "match weight" is currently reduced by at least attenuate_weight_regain, variing with an error function down to 0 if the track is
          // inside the chamber.
          // remark: the additional scale of 0.5 normalizes Err to run from 0 to 1 in y
          station_weight[i - 1] = station_weight[i - 1] * attenuate_weight_regain * 0.5f *
                                  (std::erf(stations_w_track_at_boundary[i - 1] / 6.f) + 1.f);
        } else if (station_has_segmentmatch[i - 1] <= 0 &&
                   stations_w_track_at_boundary[i - 1] == 0.f) {  // no segment match and track well inside chamber
          // full penalization
          station_weight[i - 1] = 0.f;
        }
      } else {  // always fully penalize tracks with no matching segment, whether the segment is close to the boundary or not.
        if (station_has_segmentmatch[i - 1] <= 0)
          station_weight[i - 1] = 0.f;
      }

      // if track has matching segment, but the matching is not high quality, penalize
      if (station_has_segmentmatch[i - 1] > 0) {
        if (i <= 4) {  // we are in the DTs
          if (muon.dY(i, 1, arbitrationType) < 999999.f &&
              muon.dX(i, 1, arbitrationType) < 999999.f) {  // have both X and Y match
            const float pullTot2 =
                std::pow(muon.pullX(i, 1, arbitrationType), 2.) + std::pow(muon.pullY(i, 1, arbitrationType), 2.);
            if (pullTot2 > 1.f) {
              const float dxy2 =
                  std::pow(muon.dX(i, 1, arbitrationType), 2.) + std::pow(muon.dY(i, 1, arbitrationType), 2.);
              // reduce weight
              if (use_match_dist_penalty) {
                // only use pull if 3 sigma is not smaller than 3 cm
                if (dxy2 < 9.f && pullTot2 > 9.f) {
                  if (dxy2 > 1.f)
                    station_weight[i - 1] *= 1.f / std::pow(dxy2, .125);
                } else {
                  station_weight[i - 1] *= 1.f / std::pow(pullTot2, .125);
                }
              }
            }
          } else if (muon.dY(i, 1, arbitrationType) >= 999999.f) {  // has no match in Y
            // has a match in X. Pull larger that 1 to avoid increasing the weight (just penalize, don't anti-penalize)
            if (muon.pullX(i, 1, arbitrationType) > 1.f) {
              // reduce weight
              if (use_match_dist_penalty) {
                // only use pull if 3 sigma is not smaller than 3 cm
                if (muon.dX(i, 1, arbitrationType) < 3.f && muon.pullX(i, 1, arbitrationType) > 3.f) {
                  if (muon.dX(i, 1, arbitrationType) > 1.f)
                    station_weight[i - 1] *= 1.f / std::pow(muon.dX(i, 1, arbitrationType), .25);
                } else {
                  station_weight[i - 1] *= 1.f / std::pow(muon.pullX(i, 1, arbitrationType), .25);
                }
              }
            }
          } else {  // has no match in X
            // has a match in Y. Pull larger that 1 to avoid increasing the weight (just penalize, don't anti-penalize)
            if (muon.pullY(i, 1, arbitrationType) > 1.f) {
              // reduce weight
              if (use_match_dist_penalty) {
                // only use pull if 3 sigma is not smaller than 3 cm
                if (muon.dY(i, 1, arbitrationType) < 3. && muon.pullY(i, 1, arbitrationType) > 3.) {
                  if (muon.dY(i, 1, arbitrationType) > 1.f)
                    station_weight[i - 1] *= 1.f / std::pow(muon.dY(i, 1, arbitrationType), .25);
                } else {
                  station_weight[i - 1] *= 1.f / std::pow(muon.pullY(i, 1, arbitrationType), .25);
                }
              }
            }
          }
        } else {  // We are in the CSCs
          const float pullTot2 =
              std::pow(muon.pullX(i - 4, 2, arbitrationType), 2.) + std::pow(muon.pullY(i - 4, 2, arbitrationType), 2.);
          if (pullTot2 > 1.f) {
            // reduce weight
            if (use_match_dist_penalty) {
              const float dxy2 =
                  std::pow(muon.dX(i - 4, 2, arbitrationType), 2.) + std::pow(muon.dY(i - 4, 2, arbitrationType), 2.);
              // only use pull if 3 sigma is not smaller than 3 cm
              if (dxy2 < 9.f && pullTot2 > 9.f) {
                if (dxy2 > 1.f)
                  station_weight[i - 1] *= 1.f / std::pow(dxy2, .125);
              } else {
                station_weight[i - 1] *= 1.f / std::pow(pullTot2, .125);
              }
            }
          }
        }
      }

      // Thoughts:
      // - should penalize if the segment has only x OR y info
      // - should also use the segment direction, as it now works!

    } else {  // track did not pass a chamber in this station - just reset weight
      station_weight[i - 1] = 0.f;
    }

    //increment final weight for muon:
    full_weight += station_weight[i - 1];
  }

  // if we don't expect any matches, we set the compatibility to
  // 0.5 as the track is as compatible with a muon as it is with
  // background - we should maybe rather set it to -0.5!
  if (nr_of_stations_crossed == 0) {
    //      full_weight = attenuate_weight_regain*0.5;
    full_weight = 0.5f;
  }

  // ********************************************************;
  // *** calculate weights for each station *****************;
  // ************** end *************************************;

  return full_weight;
}

bool muon::isGoodMuon(const reco::Muon& muon,
                      AlgorithmType type,
                      double minCompatibility,
                      reco::Muon::ArbitrationType arbitrationType) {
  if (!muon.isMatchesValid())
    return false;
  bool goodMuon = false;

  switch (type) {
    case TM2DCompatibility:
      // Simplistic first cut in the 2D segment- vs calo-compatibility plane. Will have to be refined!
      if (((0.8 * caloCompatibility(muon)) + (1.2 * segmentCompatibility(muon, arbitrationType))) > minCompatibility)
        goodMuon = true;
      else
        goodMuon = false;
      return goodMuon;
      break;
    default:
      // 	LogTrace("MuonIdentification")<<"            // Invalid Algorithm Type called!";
      goodMuon = false;
      return goodMuon;
      break;
  }
}

bool muon::isGoodMuon(const reco::Muon& muon,
                      AlgorithmType type,
                      int minNumberOfMatches,
                      double maxAbsDx,
                      double maxAbsPullX,
                      double maxAbsDy,
                      double maxAbsPullY,
                      double maxChamberDist,
                      double maxChamberDistPull,
                      reco::Muon::ArbitrationType arbitrationType,
                      bool syncMinNMatchesNRequiredStationsInBarrelOnly,
                      bool applyAlsoAngularCuts) {
  if (!muon.isMatchesValid())
    return false;
  bool goodMuon = false;

  if (type == TMLastStation) {
    // To satisfy my own paranoia, if the user specifies that the
    // minimum number of matches is zero, then return true.
    if (minNumberOfMatches == 0)
      return true;

    unsigned int theStationMask = muon.stationMask(arbitrationType);
    unsigned int theRequiredStationMask =
        RequiredStationMask(muon, maxChamberDist, maxChamberDistPull, arbitrationType);

    // Require that there be at least a minimum number of segments
    int numSegs = 0;
    int numRequiredStations = 0;
    for (int it = 0; it < 8; ++it) {
      if (theStationMask & 1 << it)
        ++numSegs;
      if (theRequiredStationMask & 1 << it)
        ++numRequiredStations;
    }

    // Make sure the minimum number of matches is not greater than
    // the number of required stations but still greater than zero
    if (syncMinNMatchesNRequiredStationsInBarrelOnly) {
      // Note that we only do this in the barrel region!
      if (std::abs(muon.eta()) < 1.2) {
        if (minNumberOfMatches > numRequiredStations)
          minNumberOfMatches = numRequiredStations;
        if (minNumberOfMatches < 1)  //SK: this only happens for negative values
          minNumberOfMatches = 1;
      }
    } else {
      if (minNumberOfMatches > numRequiredStations)
        minNumberOfMatches = numRequiredStations;
      if (minNumberOfMatches < 1)  //SK: this only happens for negative values
        minNumberOfMatches = 1;
    }

    if (numSegs >= minNumberOfMatches)
      goodMuon = true;

    // Require that last required station have segment
    // If there are zero required stations keep track
    // of the last station with a segment so that we may
    // apply the quality cuts below to it instead
    int lastSegBit = 0;
    if (theRequiredStationMask) {
      for (int stationIdx = 7; stationIdx >= 0; --stationIdx)
        if (theRequiredStationMask & 1 << stationIdx) {
          if (theStationMask & 1 << stationIdx) {
            lastSegBit = stationIdx;
            goodMuon &= 1;
            break;
          } else {
            goodMuon = false;
            break;
          }
        }
    } else {
      for (int stationIdx = 7; stationIdx >= 0; --stationIdx)
        if (theStationMask & 1 << stationIdx) {
          lastSegBit = stationIdx;
          break;
        }
    }

    if (!goodMuon)
      return false;

    // Impose pull cuts on last segment
    int station = 0, detector = 0;
    station = lastSegBit < 4 ? lastSegBit + 1 : lastSegBit - 3;
    detector = lastSegBit < 4 ? 1 : 2;

    // Check x information
    if (std::abs(muon.pullX(station, detector, arbitrationType, true)) > maxAbsPullX &&
        std::abs(muon.dX(station, detector, arbitrationType)) > maxAbsDx)
      return false;

    if (applyAlsoAngularCuts && std::abs(muon.pullDxDz(station, detector, arbitrationType, true)) > maxAbsPullX)
      return false;

    // Is this a tight algorithm, i.e. do we bother to check y information?
    if (maxAbsDy < 999999) {  // really if maxAbsDy < 1E9 as currently defined

      // Check y information
      if (detector == 2) {  // CSC
        if (std::abs(muon.pullY(station, 2, arbitrationType, true)) > maxAbsPullY &&
            std::abs(muon.dY(station, 2, arbitrationType)) > maxAbsDy)
          return false;

        if (applyAlsoAngularCuts && std::abs(muon.pullDyDz(station, 2, arbitrationType, true)) > maxAbsPullY)
          return false;
      } else {
        //
        // In DT, if this is a "Tight" algorithm and the last segment is
        // missing y information (always the case in station 4!!!), impose
        // respective cuts on the next station in the stationMask that has
        // a segment with y information.  If there are no segments with y
        // information then there is nothing to penalize. Should we
        // penalize in Tight for having zero segments with y information?
        // That is the fundamental question.  Of course I am being uber
        // paranoid; if this is a good muon then there will probably be at
        // least one segment with y information but not always.  Suppose
        // somehow a muon only creates segments in station 4, then we
        // definitely do not want to require that there be at least one
        // segment with y information because we will lose it completely.
        //

        for (int stationIdx = station; stationIdx > 0; --stationIdx) {
          if (!(theStationMask & 1 << (stationIdx - 1)))  // don't bother if the station is not in the stationMask
            continue;

          if (muon.dY(stationIdx, 1, arbitrationType) > 999998)  // no y-information
            continue;

          if (std::abs(muon.pullY(stationIdx, 1, arbitrationType, true)) > maxAbsPullY &&
              std::abs(muon.dY(stationIdx, 1, arbitrationType)) > maxAbsDy) {
            return false;
          }

          if (applyAlsoAngularCuts && std::abs(muon.pullDyDz(stationIdx, 1, arbitrationType, true)) > maxAbsPullY)
            return false;

          // If we get this far then great this is a good muon
          return true;
        }
      }
    }

    return goodMuon;
  }  // TMLastStation

  // TMOneStation requires only that there be one "good" segment, regardless
  // of the required stations.  We do not penalize if there are absolutely zero
  // segments with y information in the Tight algorithm.  Maybe I'm being
  // paranoid but so be it.  If it's really a good muon then we will probably
  // find at least one segment with both x and y information but you never
  // know, and I don't want to deal with a potential inefficiency in the DT
  // like we did with the original TMLastStation.  Incidentally, not penalizing
  // for total lack of y information in the Tight algorithm is what is done in
  // the new TMLastStation
  //
  if (type == TMOneStation) {
    unsigned int theStationMask = muon.stationMask(arbitrationType);

    // Of course there must be at least one segment
    if (!theStationMask)
      return false;

    int station = 0, detector = 0;
    // Keep track of whether or not there is a DT segment with y information.
    // In the end, if it turns out there are absolutely zero DT segments with
    // y information, require only that there was a segment with good x info.
    // This of course only applies to the Tight algorithms.
    bool existsGoodDTSegX = false;
    bool existsDTSegY = false;

    // Impose cuts on the segments in the station mask until we find a good one
    // Might as well start with the lowest bit to speed things up.
    for (int stationIdx = 0; stationIdx <= 7; ++stationIdx)
      if (theStationMask & 1 << stationIdx) {
        station = stationIdx < 4 ? stationIdx + 1 : stationIdx - 3;
        detector = stationIdx < 4 ? 1 : 2;

        if ((std::abs(muon.pullX(station, detector, arbitrationType, true)) > maxAbsPullX &&
             std::abs(muon.dX(station, detector, arbitrationType)) > maxAbsDx) ||
            (applyAlsoAngularCuts && std::abs(muon.pullDxDz(station, detector, arbitrationType, true)) > maxAbsPullX))
          continue;
        else if (detector == 1)
          existsGoodDTSegX = true;

        // Is this a tight algorithm?  If yes, use y information
        if (maxAbsDy < 999999) {
          if (detector == 2) {  // CSC
            if ((std::abs(muon.pullY(station, 2, arbitrationType, true)) > maxAbsPullY &&
                 std::abs(muon.dY(station, 2, arbitrationType)) > maxAbsDy) ||
                (applyAlsoAngularCuts && std::abs(muon.pullDyDz(station, 2, arbitrationType, true)) > maxAbsPullY))
              continue;
          } else {
            if (muon.dY(station, 1, arbitrationType) > 999998)  // no y-information
              continue;
            else
              existsDTSegY = true;

            if ((std::abs(muon.pullY(station, 1, arbitrationType, true)) > maxAbsPullY &&
                 std::abs(muon.dY(station, 1, arbitrationType)) > maxAbsDy) ||
                (applyAlsoAngularCuts && std::abs(muon.pullDyDz(station, 1, arbitrationType, true)) > maxAbsPullY)) {
              continue;
            }
          }
        }

        // If we get this far then great this is a good muon
        return true;
      }

    // If we get this far then for sure there are no "good" CSC segments. For
    // DT, check if there were any segments with y information.  If there
    // were none, but there was a segment with good x, then we're happy. If
    // there WERE segments with y information, then they must have been shit
    // since we are here so fail it.  Of course, if this is a Loose algorithm
    // then fail immediately since if we had good x we would already have
    // returned true
    if (maxAbsDy < 999999) {
      if (existsDTSegY)
        return false;
      else if (existsGoodDTSegX)
        return true;
    } else
      return false;
  }  // TMOneStation

  if (type == RPCMu) {
    if (minNumberOfMatches == 0)
      return true;

    int nMatch = 0;
    for (const auto& chamberMatch : muon.matches()) {
      if (chamberMatch.detector() != MuonSubdetId::RPC)
        continue;

      const float trkX = chamberMatch.x;
      const float errX = chamberMatch.xErr;

      for (const auto& rpcMatch : chamberMatch.rpcMatches) {
        const float rpcX = rpcMatch.x;
        const float dX = std::abs(rpcX - trkX);
        if (dX < maxAbsDx or dX < maxAbsPullX * errX) {
          ++nMatch;
          break;
        }
      }
    }

    if (nMatch >= minNumberOfMatches)
      return true;
    else
      return false;
  }  // RPCMu

  if (type == ME0Mu) {
    if (minNumberOfMatches == 0)
      return true;

    int nMatch = 0;
    for (const auto& chamberMatch : muon.matches()) {
      if (chamberMatch.detector() != MuonSubdetId::ME0)
        continue;

      const float trkX = chamberMatch.x;
      const float errX2 = chamberMatch.xErr * chamberMatch.xErr;
      const float trkY = chamberMatch.y;
      const float errY2 = chamberMatch.yErr * chamberMatch.yErr;

      for (const auto& segment : chamberMatch.me0Matches) {
        const float me0X = segment.x;
        const float me0ErrX2 = segment.xErr * segment.xErr;
        const float me0Y = segment.y;
        const float me0ErrY2 = segment.yErr * segment.yErr;

        const float dX = std::abs(me0X - trkX);
        const float dY = std::abs(me0Y - trkY);
        const float invPullX2 = errX2 + me0ErrX2;
        const float invPullY2 = errY2 + me0ErrY2;

        if ((dX < maxAbsDx or dX < maxAbsPullX * std::sqrt(invPullX2)) and
            (dY < maxAbsDy or dY < maxAbsPullY * std::sqrt(invPullY2))) {
          ++nMatch;
          break;
        }
      }
    }

    return (nMatch >= minNumberOfMatches);
  }  // ME0Mu

  if (type == GEMMu) {
    if (minNumberOfMatches == 0)
      return true;

    int nMatch = 0;
    for (const auto& chamberMatch : muon.matches()) {
      if (chamberMatch.detector() != MuonSubdetId::GEM)
        continue;

      const float trkX = chamberMatch.x;
      const float errX2 = chamberMatch.xErr * chamberMatch.xErr;
      const float trkY = chamberMatch.y;
      const float errY2 = chamberMatch.yErr * chamberMatch.yErr;

      for (const auto& segment : chamberMatch.gemMatches) {
        const float gemX = segment.x;
        const float gemErrX2 = segment.xErr * segment.xErr;
        const float gemY = segment.y;
        const float gemErrY2 = segment.yErr * segment.yErr;

        const float dX = std::abs(gemX - trkX);
        const float dY = std::abs(gemY - trkY);
        const float invPullX2 = errX2 + gemErrX2;
        const float invPullY2 = errY2 + gemErrY2;

        if ((dX < maxAbsDx or dX < maxAbsPullX * std::sqrt(invPullX2)) and
            (dY < maxAbsDy or dY < maxAbsPullY * std::sqrt(invPullY2))) {
          ++nMatch;
          break;
        }
      }
    }

    return (nMatch >= minNumberOfMatches);
  }  // GEMMu

  return goodMuon;
}

bool muon::isGoodMuon(const reco::Muon& muon, SelectionType type, reco::Muon::ArbitrationType arbitrationType) {
  switch (type) {
    case muon::All:
      return true;
      break;
    case muon::AllGlobalMuons:
      return muon.isGlobalMuon();
      break;
    case muon::AllTrackerMuons:
      return muon.isTrackerMuon();
      break;
    case muon::AllStandAloneMuons:
      return muon.isStandAloneMuon();
      break;
    case muon::TrackerMuonArbitrated:
      return muon.isTrackerMuon() && muon.numberOfMatches(arbitrationType) > 0;
      break;
    case muon::AllArbitrated:
      return !muon.isTrackerMuon() || muon.numberOfMatches(arbitrationType) > 0;
      break;
    case muon::GlobalMuonPromptTight:
      return muon.isGlobalMuon() && muon.globalTrack()->normalizedChi2() < 10. &&
             muon.globalTrack()->hitPattern().numberOfValidMuonHits() > 0;
      break;
      // For "Loose" algorithms we choose maximum y quantity cuts of 1E9 instead of
      // 9999 as before.  We do this because the muon methods return 999999 (note
      // there are six 9's) when the requested information is not available.  For
      // example, if a muon fails to traverse the z measuring superlayer in a station
      // in the DT, then all methods involving segmentY in this station return
      // 999999 to demonstrate that the information is missing.  In order to not
      // penalize muons for missing y information in Loose algorithms where we do
      // not care at all about y information, we raise these limits.  In the
      // TMLastStation and TMOneStation algorithms we actually use this huge number
      // to determine whether to consider y information at all.
    case muon::TMLastStationLoose:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMLastStation, 2, 3, 3, 1E9, 1E9, -3, -3, arbitrationType, true, false);
      break;
    case muon::TMLastStationTight:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMLastStation, 2, 3, 3, 3, 3, -3, -3, arbitrationType, true, false);
      break;
    case muon::TMOneStationLoose:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMOneStation, 1, 3, 3, 1E9, 1E9, 1E9, 1E9, arbitrationType, false, false);
      break;
    case muon::TMOneStationTight:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMOneStation, 1, 3, 3, 3, 3, 1E9, 1E9, arbitrationType, false, false);
      break;
    case muon::TMLastStationOptimizedLowPtLoose:
      if (muon.pt() < 8. && std::abs(muon.eta()) < 1.2)
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMOneStation, 1, 3, 3, 1E9, 1E9, 1E9, 1E9, arbitrationType, false, false);
      else
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMLastStation, 2, 3, 3, 1E9, 1E9, -3, -3, arbitrationType, false, false);
      break;
    case muon::TMLastStationOptimizedLowPtTight:
      if (muon.pt() < 8. && std::abs(muon.eta()) < 1.2)
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMOneStation, 1, 3, 3, 3, 3, 1E9, 1E9, arbitrationType, false, false);
      else
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMLastStation, 2, 3, 3, 3, 3, -3, -3, arbitrationType, false, false);
      break;
      //compatibility loose
    case muon::TM2DCompatibilityLoose:
      return muon.isTrackerMuon() && isGoodMuon(muon, TM2DCompatibility, 0.7, arbitrationType);
      break;
      //compatibility tight
    case muon::TM2DCompatibilityTight:
      return muon.isTrackerMuon() && isGoodMuon(muon, TM2DCompatibility, 1.0, arbitrationType);
      break;
    case muon::GMTkChiCompatibility:
      return muon.isGlobalMuon() && muon.isQualityValid() &&
             std::abs(muon.combinedQuality().trkRelChi2 - muon.innerTrack()->normalizedChi2()) < 2.0;
      break;
    case muon::GMStaChiCompatibility:
      return muon.isGlobalMuon() && muon.isQualityValid() &&
             std::abs(muon.combinedQuality().staRelChi2 - muon.outerTrack()->normalizedChi2()) < 2.0;
      break;
    case muon::GMTkKinkTight:
      return muon.isGlobalMuon() && muon.isQualityValid() && muon.combinedQuality().trkKink < 100.0;
      break;
    case muon::TMLastStationAngLoose:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMLastStation, 2, 3, 3, 1E9, 1E9, -3, -3, arbitrationType, false, true);
      break;
    case muon::TMLastStationAngTight:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMLastStation, 2, 3, 3, 3, 3, -3, -3, arbitrationType, false, true);
      break;
    case muon::TMOneStationAngLoose:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMOneStation, 1, 3, 3, 1E9, 1E9, 1E9, 1E9, arbitrationType, false, true);
      break;
    case muon::TMOneStationAngTight:
      return muon.isTrackerMuon() &&
             isGoodMuon(muon, TMOneStation, 1, 3, 3, 3, 3, 1E9, 1E9, arbitrationType, false, true);
      break;
    case muon::TMLastStationOptimizedBarrelLowPtLoose:
      if (muon.pt() < 8. && std::abs(muon.eta()) < 1.2)
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMOneStation, 1, 3, 3, 1E9, 1E9, 1E9, 1E9, arbitrationType, false, false);
      else
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMLastStation, 2, 3, 3, 1E9, 1E9, -3, -3, arbitrationType, true, false);
      break;
    case muon::TMLastStationOptimizedBarrelLowPtTight:
      if (muon.pt() < 8. && std::abs(muon.eta()) < 1.2)
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMOneStation, 1, 3, 3, 3, 3, 1E9, 1E9, arbitrationType, false, false);
      else
        return muon.isTrackerMuon() &&
               isGoodMuon(muon, TMLastStation, 2, 3, 3, 3, 3, -3, -3, arbitrationType, true, false);
      break;
    case muon::RPCMuLoose:
      return muon.isRPCMuon() && isGoodMuon(muon, RPCMu, 2, 20, 4, 1e9, 1e9, 1e9, 1e9, arbitrationType, false, false);
      break;
    case muon::AllME0Muons:
      return muon.isME0Muon();
      break;
    case muon::ME0MuonArbitrated:
      return muon.isME0Muon() &&
             isGoodMuon(muon, ME0Mu, 1, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, arbitrationType, false, false);
      break;
    case muon::AllGEMMuons:
      return muon.isGEMMuon();
      break;
    case muon::GEMMuonArbitrated:
      return muon.isGEMMuon() &&
             isGoodMuon(muon, GEMMu, 1, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, arbitrationType, false, false);
      break;
    case muon::TriggerIdLoose:
      return isLooseTriggerMuon(muon);
      break;
    default:
      return false;
  }
}

bool muon::overlap(
    const reco::Muon& muon1, const reco::Muon& muon2, double pullX, double pullY, bool checkAdjacentChambers) {
  unsigned int nMatches1 = muon1.numberOfMatches(reco::Muon::SegmentAndTrackArbitration);
  unsigned int nMatches2 = muon2.numberOfMatches(reco::Muon::SegmentAndTrackArbitration);
  unsigned int betterMuon = (muon1.pt() > muon2.pt() ? 1 : 2);
  for (std::vector<reco::MuonChamberMatch>::const_iterator chamber1 = muon1.matches().begin();
       chamber1 != muon1.matches().end();
       ++chamber1)
    for (std::vector<reco::MuonChamberMatch>::const_iterator chamber2 = muon2.matches().begin();
         chamber2 != muon2.matches().end();
         ++chamber2) {
      // if ( (chamber1->segmentMatches.empty() || chamber2->segmentMatches.empty()) ) continue;

      // handle case where both muons have information about the same chamber
      // here we know how close they are
      if (chamber1->id == chamber2->id) {
        // found the same chamber
        if (std::abs(chamber1->x - chamber2->x) <
            pullX * sqrt(chamber1->xErr * chamber1->xErr + chamber2->xErr * chamber2->xErr)) {
          if (betterMuon == 1)
            nMatches2--;
          else
            nMatches1--;
          if (nMatches1 == 0 || nMatches2 == 0)
            return true;
          continue;
        }
        if (std::abs(chamber1->y - chamber2->y) <
            pullY * sqrt(chamber1->yErr * chamber1->yErr + chamber2->yErr * chamber2->yErr)) {
          if (betterMuon == 1)
            nMatches2--;
          else
            nMatches1--;
          if (nMatches1 == 0 || nMatches2 == 0)
            return true;
        }
      } else {
        if (!checkAdjacentChambers)
          continue;
        // check if tracks are pointing into overlaping region of the CSC detector
        if (chamber1->id.subdetId() != MuonSubdetId::CSC || chamber2->id.subdetId() != MuonSubdetId::CSC)
          continue;
        CSCDetId id1(chamber1->id);
        CSCDetId id2(chamber2->id);
        if (id1.endcap() != id2.endcap())
          continue;
        if (id1.station() != id2.station())
          continue;
        if (id1.ring() != id2.ring())
          continue;
        if (std::abs(id1.chamber() - id2.chamber()) > 1)
          continue;
        // FIXME: we don't handle 18->1; 36->1 transitions since
        // I don't know how to check for sure how many chambers
        // are there. Probably need to hard code some checks.

        // Now we have to make sure that both tracks are close to an edge
        // FIXME: ignored Y coordinate for now
        if (std::abs(chamber1->edgeX) > chamber1->xErr * pullX)
          continue;
        if (std::abs(chamber2->edgeX) > chamber2->xErr * pullX)
          continue;
        if (chamber1->x * chamber2->x < 0) {  // check if the same edge
          if (betterMuon == 1)
            nMatches2--;
          else
            nMatches1--;
          if (nMatches1 == 0 || nMatches2 == 0)
            return true;
        }
      }
    }
  return false;
}

bool muon::isLooseTriggerMuon(const reco::Muon& muon) {
  // Requirements:
  // - no depencence on information not availabe in the muon object
  // - use only robust inputs
  bool tk_id = muon::isGoodMuon(muon, TMOneStationTight);
  if (not tk_id)
    return false;
  bool layer_requirements = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                            muon.innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0;
  bool match_requirements =
      (muon.expectedNnumberOfMatchedStations() < 2) or (muon.numberOfMatchedStations() > 1) or (muon.pt() < 8);
  return layer_requirements and match_requirements;
}

bool muon::isTightMuon(const reco::Muon& muon, const reco::Vertex& vtx) {
  if (!muon.isPFMuon() || !muon.isGlobalMuon())
    return false;

  bool muID = isGoodMuon(muon, GlobalMuonPromptTight) && (muon.numberOfMatchedStations() > 1);

  bool hits = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
              muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

  bool ip = std::abs(muon.muonBestTrack()->dxy(vtx.position())) < 0.2 &&
            std::abs(muon.muonBestTrack()->dz(vtx.position())) < 0.5;

  return muID && hits && ip;
}

bool muon::isLooseMuon(const reco::Muon& muon) {
  return muon.isPFMuon() && (muon.isGlobalMuon() || muon.isTrackerMuon());
}

bool muon::isMediumMuon(const reco::Muon& muon, bool run2016_hip_mitigation) {
  if (not isLooseMuon(muon))
    return false;
  if (run2016_hip_mitigation) {
    if (muon.innerTrack()->validFraction() < 0.49)
      return false;
  } else {
    if (muon.innerTrack()->validFraction() < 0.8)
      return false;
  }

  bool goodGlb = muon.isGlobalMuon() && muon.globalTrack()->normalizedChi2() < 3. &&
                 muon.combinedQuality().chi2LocalPosition < 12. && muon.combinedQuality().trkKink < 20.;

  return (segmentCompatibility(muon) > (goodGlb ? 0.303 : 0.451));
}

bool muon::isSoftMuon(const reco::Muon& muon, const reco::Vertex& vtx, bool run2016_hip_mitigation) {
  bool muID = muon::isGoodMuon(muon, TMOneStationTight);

  if (!muID)
    return false;

  bool layers = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                muon.innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0;

  bool ishighq = muon.innerTrack()->quality(reco::Track::highPurity);

  bool ip =
      std::abs(muon.innerTrack()->dxy(vtx.position())) < 0.3 && std::abs(muon.innerTrack()->dz(vtx.position())) < 20.;

  return layers && ip && (ishighq | run2016_hip_mitigation);
}

bool muon::isHighPtMuon(const reco::Muon& muon, const reco::Vertex& vtx) {
  if (!muon.isGlobalMuon())
    return false;

  bool muValHits = (muon.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 ||
                    muon.tunePMuonBestTrack()->hitPattern().numberOfValidMuonHits() > 0);

  bool muMatchedSt = muon.numberOfMatchedStations() > 1;
  if (!muMatchedSt) {
    if (muon.isTrackerMuon() && muon.numberOfMatchedStations() == 1) {
      if (muon.expectedNnumberOfMatchedStations() < 2 || !(muon.stationMask() == 1 || muon.stationMask() == 16) ||
          muon.numberOfMatchedRPCLayers() > 2)
        muMatchedSt = true;
    }
  }

  bool muID = muValHits && muMatchedSt;

  bool hits = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
              muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

  bool momQuality = muon.tunePMuonBestTrack()->ptError() / muon.tunePMuonBestTrack()->pt() < 0.3;

  bool ip =
      std::abs(muon.innerTrack()->dxy(vtx.position())) < 0.2 && std::abs(muon.innerTrack()->dz(vtx.position())) < 0.5;

  return muID && hits && momQuality && ip;
}

bool muon::isTrackerHighPtMuon(const reco::Muon& muon, const reco::Vertex& vtx) {
  bool muID = muon.isTrackerMuon() && muon.track().isNonnull() && (muon.numberOfMatchedStations() > 1);
  if (!muID)
    return false;

  bool hits = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
              muon.innerTrack()->hitPattern().numberOfValidPixelHits() > 0;

  bool momQuality = muon.tunePMuonBestTrack()->ptError() < 0.3 * muon.tunePMuonBestTrack()->pt();

  bool ip =
      std::abs(muon.innerTrack()->dxy(vtx.position())) < 0.2 && std::abs(muon.innerTrack()->dz(vtx.position())) < 0.5;

  return muID && hits && momQuality && ip;
}

int muon::sharedSegments(const reco::Muon& mu, const reco::Muon& mu2, unsigned int segmentArbitrationMask) {
  int ret = 0;

  // Will do with a stupid double loop, since creating and filling a map is probably _more_ inefficient for a single lookup.
  for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = mu.matches().begin();
       chamberMatch != mu.matches().end();
       ++chamberMatch) {
    if (chamberMatch->segmentMatches.empty())
      continue;
    for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch2 = mu2.matches().begin();
         chamberMatch2 != mu2.matches().end();
         ++chamberMatch2) {
      if (chamberMatch2->segmentMatches.empty())
        continue;
      if (chamberMatch2->id() != chamberMatch->id())
        continue;
      for (std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
           segmentMatch != chamberMatch->segmentMatches.end();
           ++segmentMatch) {
        if (!segmentMatch->isMask(segmentArbitrationMask))
          continue;
        for (std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch2 = chamberMatch2->segmentMatches.begin();
             segmentMatch2 != chamberMatch2->segmentMatches.end();
             ++segmentMatch2) {
          if (!segmentMatch2->isMask(segmentArbitrationMask))
            continue;
          if ((segmentMatch->cscSegmentRef.isNonnull() &&
               segmentMatch->cscSegmentRef == segmentMatch2->cscSegmentRef) ||
              (segmentMatch->dtSegmentRef.isNonnull() && segmentMatch->dtSegmentRef == segmentMatch2->dtSegmentRef)) {
            ++ret;
          }  // is the same
        }    // segment of mu2 in chamber
      }      // segment of mu1 in chamber
    }        // chamber of mu2
  }          // chamber of mu1

  return ret;
}

bool outOfTimeMuon(const reco::Muon& muon) {
  const auto& combinedTime = muon.time();
  const auto& rpcTime = muon.rpcTime();
  bool combinedTimeIsOk = (combinedTime.nDof > 7);
  bool rpcTimeIsOk = (rpcTime.nDof > 1 && std::abs(rpcTime.timeAtIpInOutErr) < 0.001);
  bool outOfTime = false;
  if (rpcTimeIsOk) {
    if ((std::abs(rpcTime.timeAtIpInOut) > 10) && !(combinedTimeIsOk && std::abs(combinedTime.timeAtIpInOut) < 10))
      outOfTime = true;
  } else {
    if (combinedTimeIsOk && (combinedTime.timeAtIpInOut > 20 || combinedTime.timeAtIpInOut < -45))
      outOfTime = true;
  }
  return outOfTime;
}

reco::Muon::Selector muon::makeSelectorBitset(reco::Muon const& muon,
                                              reco::Vertex const* vertex,
                                              bool run2016_hip_mitigation) {
  // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
  unsigned int selectors = muon.selectors();
  // Compute Id and Isolation variables
  double chIso = muon.pfIsolationR04().sumChargedHadronPt;
  double nIso = muon.pfIsolationR04().sumNeutralHadronEt;
  double phoIso = muon.pfIsolationR04().sumPhotonEt;
  double puIso = muon.pfIsolationR04().sumPUPt;
  double dbCorrectedIsolation = chIso + std::max(nIso + phoIso - .5 * puIso, 0.);
  double dbCorrectedRelIso = dbCorrectedIsolation / muon.pt();
  double tkRelIso = muon.isolationR03().sumPt / muon.pt();

  // Base selectors
  if (muon::isLooseMuon(muon))
    selectors |= reco::Muon::CutBasedIdLoose;
  if (vertex) {
    if (muon::isTightMuon(muon, *vertex))
      selectors |= reco::Muon::CutBasedIdTight;
    if (muon::isSoftMuon(muon, *vertex, run2016_hip_mitigation))
      selectors |= reco::Muon::SoftCutBasedId;
    if (muon::isHighPtMuon(muon, *vertex))
      selectors |= reco::Muon::CutBasedIdGlobalHighPt;
    if (muon::isTrackerHighPtMuon(muon, *vertex))
      selectors |= reco::Muon::CutBasedIdTrkHighPt;
  }
  if (muon::isMediumMuon(muon, run2016_hip_mitigation)) {
    selectors |= reco::Muon::CutBasedIdMedium;
    if (vertex and std::abs(muon.muonBestTrack()->dz(vertex->position())) < 0.1 and
        std::abs(muon.muonBestTrack()->dxy(vertex->position())) < 0.02)
      selectors |= reco::Muon::CutBasedIdMediumPrompt;
  }

  // PF isolation
  if (dbCorrectedRelIso < 0.40)
    selectors |= reco::Muon::PFIsoVeryLoose;
  if (dbCorrectedRelIso < 0.25)
    selectors |= reco::Muon::PFIsoLoose;
  if (dbCorrectedRelIso < 0.20)
    selectors |= reco::Muon::PFIsoMedium;
  if (dbCorrectedRelIso < 0.15)
    selectors |= reco::Muon::PFIsoTight;
  if (dbCorrectedRelIso < 0.10)
    selectors |= reco::Muon::PFIsoVeryTight;
  if (dbCorrectedRelIso < 0.05)
    selectors |= reco::Muon::PFIsoVeryVeryTight;

  // Tracker isolation
  if (tkRelIso < 0.10)
    selectors |= reco::Muon::TkIsoLoose;
  if (tkRelIso < 0.05)
    selectors |= reco::Muon::TkIsoTight;

  // Trigger selectors
  if (isLooseTriggerMuon(muon))
    selectors |= reco::Muon::TriggerIdLoose;

  // Timing
  if (!outOfTimeMuon(muon))
    selectors |= reco::Muon::InTimeMuon;

  return static_cast<reco::Muon::Selector>(selectors);
}
