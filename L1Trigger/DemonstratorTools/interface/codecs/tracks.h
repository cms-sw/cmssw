
#ifndef L1Trigger_DemonstratorTools_codecs_tracks_h
#define L1Trigger_DemonstratorTools_codecs_tracks_h

#include <array>
#include <sstream>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t::demo::codecs {
  //function to get the gttLinkID from the TrackFindingProcessors
  template <typename T>
  unsigned int gttLinkID(T track) {
    // use the sign bit of the tanL word to remove dependence on TTTrack eta member.
    unsigned int etaSector = (track.getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kTanlMSB,
                                                   TTTrack_TrackWord::TrackBitLocations::kTanlMSB)
                                  ? 0
                                  : 1);
    return etaSector + (2 * track.phiSector());
  }

  static inline std::pair<unsigned int, unsigned int> sectorsEtaPhiFromGTTLinkID(unsigned int id) {
    unsigned int etaSector = (id % 2);
    unsigned int phiSector = (static_cast<unsigned int>(id) - etaSector) / 2;
    return std::pair<unsigned int, unsigned int>(etaSector, phiSector);
  }

  // Return true if a track is contained within a collection
  bool trackInCollection(const edm::Ref<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>&,
                         const edm::Handle<edm::RefVector<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>>&);

  // Encodes a single track into a 96-bit track word
  ap_uint<96> encodeTrack(const TTTrack_TrackWord& t);

  // Return the 96-bit track words from a given track collection and place them on the appropriate 18 'logical' links
  std::array<std::vector<ap_uint<96>>, 18> getTrackWords(const edm::View<TTTrack<Ref_Phase2TrackerDigi_>>&);
  std::array<std::vector<ap_uint<96>>, 18> getTrackWords(
      const edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>&,
      const edm::Handle<edm::RefVector<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>>&);

  // Encodes track collection onto 18 'logical' output links (2x9 eta-phi sectors; -/+ eta pairs)
  std::array<std::vector<ap_uint<64>>, 18> encodeTracks(const edm::View<TTTrack<Ref_Phase2TrackerDigi_>>&,
                                                        int debug = 0);

  // Encodes a track collection based off the ordering of another track collection
  // Requirement: The second collection must be a subset of the first
  std::array<std::vector<ap_uint<64>>, 18> encodeTracks(
      const edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>&,
      const edm::Handle<edm::RefVector<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>>&,
      int debug = 0);

  // Decodes the tracks for a single link
  std::vector<TTTrack_TrackWord> decodeTracks(const std::vector<ap_uint<64>>&);

  // Decodes the tracks from 18 'logical' output links (2x9 eta-phi sectors; , -/+ eta pairs)
  std::array<std::vector<TTTrack_TrackWord>, 18> decodeTracks(const std::array<std::vector<ap_uint<64>>, 18>&);

}  // namespace l1t::demo::codecs

#endif
