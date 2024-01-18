#ifndef PHASE2GMT_TRACKCONVERTER
#define PHASE2GMT_TRACKCONVERTER

#include "L1Trigger/Phase2L1GMT/interface/ConvertedTTTrack.h"
#include "L1Trigger/Phase2L1GMT/interface/TPSLUTs.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace Phase2L1GMT {

  class TrackConverter {
  public:
    TrackConverter(const edm::ParameterSet& iConfig);
    ~TrackConverter();

    std::vector<ConvertedTTTrack> convertTracks(const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks);

  private:
    int verbose_;
    typedef ap_uint<96> wordtype;

    uint generateQuality(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& track) { return 1; }

    uint ptLookup(uint absCurv) {
      for (auto i : ptShifts) {
        if (absCurv >= uint(i[0]) && absCurv < uint(i[1])) {
          if (i[2] < 0)
            return i[4];
          else
            return (absCurv >> i[2]) + i[3];
        }
      }
      return 0;
    }

    uint etaLookup(uint absTanL) {
      for (auto i : etaShifts) {
        if (absTanL >= uint(i[0]) && absTanL < uint(i[1])) {
          if (i[2] < 0)
            return i[4];
          else
            return (absTanL >> i[2]) + i[3];
        }
      }
      return 0;
    }

    ConvertedTTTrack convert(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& track);
  };
}  // namespace Phase2L1GMT

#endif
