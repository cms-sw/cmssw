#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/tracks.h"

namespace l1t::demo::codecs {

  ap_uint<96> encodeTrack(const TTTrack_TrackWord& t) { return t.getTrackWord(); }

  // Encodes track collection onto 18 output links (2x9 eta-phi sectors; first 9 negative eta)
  std::array<std::vector<ap_uint<64>>, 18> encodeTracks(const edm::View<TTTrack<Ref_Phase2TrackerDigi_>>& tracks,
                                                        int debug) {
    std::array<std::vector<ap_uint<96>>, 18> trackWords;
    if (debug > 0) {
      edm::LogInfo("l1t::demo::codecs") << "encodeTrack::Encoding " << tracks.size() << " tracks";
    }
    for (const auto& track : tracks)
      trackWords.at((track.eta() >= 0 ? 9 : 0) + track.phiSector()).push_back(encodeTrack(track));

    std::array<std::vector<ap_uint<64>>, 18> linkData;

    size_t counter = 0;
    for (size_t i = 0; i < linkData.size(); i++) {
      // Pad track vectors -> full packet length (156 frames = 104 tracks)
      trackWords.at(i).resize(104, 0);
      linkData.at(i).resize(156, {0});

      for (size_t j = 0; (j < trackWords.at(i).size()); j += 2) {
        linkData.at(i).at(3 * j / 2) = trackWords.at(i).at(j)(63, 0);
        linkData.at(i).at(3 * j / 2 + 1) =
            (ap_uint<32>(trackWords.at(i).at(j + 1)(31, 0)), ap_uint<32>(trackWords.at(i).at(j)(95, 64)));
        linkData.at(i).at(3 * j / 2 + 2) = trackWords.at(i).at(j + 1)(95, 32);
        counter += trackWords.at(i).at(j)(95, 95) + trackWords.at(i).at(j + 1)(95, 95);
      }
    }
    if (debug > 0) {
      edm::LogInfo("l1t::demo::codecs") << "encodeTrack::Encoded " << counter << " tracks";
    }

    return linkData;
  }

  std::vector<TTTrack_TrackWord> decodeTracks(const std::vector<ap_uint<64>>& frames) {
    std::vector<TTTrack_TrackWord> tracks;

    if ((frames.size() % 3) != 0) {
      std::stringstream message;
      message << "The number of track frames (" << frames.size() << ") is not evenly divisible by 3";
      throw std::runtime_error(message.str());
    }

    for (size_t i = 0; i < frames.size(); i += 3) {
      TTTrack_TrackWord::tkword_t combination1 = (ap_uint<32>(frames.at(i + 1)(31, 0)), frames.at(i)(63, 0));
      TTTrack_TrackWord::tkword_t combination2 = (frames.at(i + 2)(63, 0), ap_uint<32>(frames.at(i + 1)(63, 32)));
      TTTrack_TrackWord track1, track2;
      track1.setTrackWord(
          TTTrack_TrackWord::valid_t(combination1(TTTrack_TrackWord::kValidMSB, TTTrack_TrackWord::kValidLSB)),
          TTTrack_TrackWord::rinv_t(combination1(TTTrack_TrackWord::kRinvMSB, TTTrack_TrackWord::kRinvLSB)),
          TTTrack_TrackWord::phi_t(combination1(TTTrack_TrackWord::kPhiMSB, TTTrack_TrackWord::kPhiLSB)),
          TTTrack_TrackWord::tanl_t(combination1(TTTrack_TrackWord::kTanlMSB, TTTrack_TrackWord::kTanlLSB)),
          TTTrack_TrackWord::z0_t(combination1(TTTrack_TrackWord::kZ0MSB, TTTrack_TrackWord::kZ0LSB)),
          TTTrack_TrackWord::d0_t(combination1(TTTrack_TrackWord::kD0MSB, TTTrack_TrackWord::kD0LSB)),
          TTTrack_TrackWord::chi2rphi_t(combination1(TTTrack_TrackWord::kChi2RPhiMSB, TTTrack_TrackWord::kChi2RPhiLSB)),
          TTTrack_TrackWord::chi2rz_t(combination1(TTTrack_TrackWord::kChi2RZMSB, TTTrack_TrackWord::kChi2RZLSB)),
          TTTrack_TrackWord::bendChi2_t(combination1(TTTrack_TrackWord::kBendChi2MSB, TTTrack_TrackWord::kBendChi2LSB)),
          TTTrack_TrackWord::hit_t(combination1(TTTrack_TrackWord::kHitPatternMSB, TTTrack_TrackWord::kHitPatternLSB)),
          TTTrack_TrackWord::qualityMVA_t(
              combination1(TTTrack_TrackWord::kMVAQualityMSB, TTTrack_TrackWord::kMVAQualityLSB)),
          TTTrack_TrackWord::otherMVA_t(
              combination1(TTTrack_TrackWord::kMVAOtherMSB, TTTrack_TrackWord::kMVAOtherLSB)));
      track2.setTrackWord(
          TTTrack_TrackWord::valid_t(combination2(TTTrack_TrackWord::kValidMSB, TTTrack_TrackWord::kValidLSB)),
          TTTrack_TrackWord::rinv_t(combination2(TTTrack_TrackWord::kRinvMSB, TTTrack_TrackWord::kRinvLSB)),
          TTTrack_TrackWord::phi_t(combination2(TTTrack_TrackWord::kPhiMSB, TTTrack_TrackWord::kPhiLSB)),
          TTTrack_TrackWord::tanl_t(combination2(TTTrack_TrackWord::kTanlMSB, TTTrack_TrackWord::kTanlLSB)),
          TTTrack_TrackWord::z0_t(combination2(TTTrack_TrackWord::kZ0MSB, TTTrack_TrackWord::kZ0LSB)),
          TTTrack_TrackWord::d0_t(combination2(TTTrack_TrackWord::kD0MSB, TTTrack_TrackWord::kD0LSB)),
          TTTrack_TrackWord::chi2rphi_t(combination2(TTTrack_TrackWord::kChi2RPhiMSB, TTTrack_TrackWord::kChi2RPhiLSB)),
          TTTrack_TrackWord::chi2rz_t(combination2(TTTrack_TrackWord::kChi2RZMSB, TTTrack_TrackWord::kChi2RZLSB)),
          TTTrack_TrackWord::bendChi2_t(combination2(TTTrack_TrackWord::kBendChi2MSB, TTTrack_TrackWord::kBendChi2LSB)),
          TTTrack_TrackWord::hit_t(combination2(TTTrack_TrackWord::kHitPatternMSB, TTTrack_TrackWord::kHitPatternLSB)),
          TTTrack_TrackWord::qualityMVA_t(
              combination2(TTTrack_TrackWord::kMVAQualityMSB, TTTrack_TrackWord::kMVAQualityLSB)),
          TTTrack_TrackWord::otherMVA_t(
              combination2(TTTrack_TrackWord::kMVAOtherMSB, TTTrack_TrackWord::kMVAOtherLSB)));
      tracks.push_back(track1);
      tracks.push_back(track2);
    }

    return tracks;
  }

  std::array<std::vector<TTTrack_TrackWord>, 18> decodeTracks(const std::array<std::vector<ap_uint<64>>, 18>& frames) {
    std::array<std::vector<TTTrack_TrackWord>, 18> tracks;

    for (size_t i = 0; i < tracks.size(); i++) {
      tracks.at(i) = decodeTracks(frames.at(i));
    }

    return tracks;
  }

}  // namespace l1t::demo::codecs
