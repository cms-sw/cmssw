#ifndef L1Trigger_TrackerTFP_TrackFindingProcessor_h
#define L1Trigger_TrackerTFP_TrackFindingProcessor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <utility>
#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to format final tfp output and to prodcue final TTTrackCollection
  class TrackFindingProcessor {
  public:
    TrackFindingProcessor(const edm::ParameterSet& iConfig,
                          const tt::Setup* setup_,
                          const DataFormats* dataFormats,
                          const TrackQuality* trackQuality);
    ~TrackFindingProcessor() {}

    // produce TTTracks
    void produce(const tt::StreamsTrack& inputs,
                 const tt::StreamsStub& stubs,
                 tt::TTTracks& ttTracks,
                 tt::StreamsTrack& outputs);
    // produce StreamsTrack
    void produce(const std::vector<TTTrackRef>& inputs, tt::StreamsTrack& outputs) const;

  private:
    //
    static constexpr int partial_width = 32;
    //
    static constexpr int partial_in = 3;
    //
    static constexpr int partial_out = 2;
    //
    typedef std::bitset<partial_width> PartialFrame;
    //
    typedef std::pair<const TTTrackRef&, PartialFrame> PartialFrameTrack;
    //
    struct Track {
      Track(const tt::FrameTrack& frameTrack,
            const tt::Frame& frameTQ,
            const std::vector<TTStubRef>& ttStubRefs,
            const TrackQuality* tq);
      const TTTrackRef& ttTrackRef_;
      const std::vector<TTStubRef> ttStubRefs_;
      bool valid_;
      std::vector<PartialFrame> partials_;
      TTBV hitPattern_;
      int channel_;
      int mva_;
      double inv2R_;
      double phiT_;
      double cot_;
      double zT_;
      double chi2rphi_;
      double chi2rz_;
    };
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    //
    void consume(const tt::StreamsTrack& inputs,
                 const tt::StreamsStub& stubs,
                 std::vector<std::deque<Track*>>& outputs);
    // emualte data format f/w
    void produce(std::vector<std::deque<Track*>>& inputs, tt::StreamsTrack& outputs) const;
    // produce TTTracks
    void produce(const tt::StreamsTrack& inputs, tt::TTTracks& ouputs) const;
    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides data formats
    const DataFormats* dataFormats_;
    // provides Track Quality algo and formats
    const TrackQuality* trackQuality_;
    // storage of tracks
    std::vector<Track> tracks_;
  };

}  // namespace trackerTFP

#endif