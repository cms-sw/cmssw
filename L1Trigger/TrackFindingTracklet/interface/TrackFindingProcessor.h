#ifndef L1Trigger_TrackFindingTracklet_TrackFindingProcessor_h
#define L1Trigger_TrackFindingTracklet_TrackFindingProcessor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <utility>
#include <vector>
#include <deque>

namespace trklet {

  /*! \class  trklet::TrackFindingProcessor
   *  \brief  Class to format final tfp output and to prodcue final TTTrackCollection
   *  \author Thomas Schuh
   *  \date   2025, Apr
   */
  class TrackFindingProcessor {
  public:
    TrackFindingProcessor(const tt::Setup* setup_,
                          const DataFormats* dataFormats,
                          const trackerTFP::TrackQuality* trackQuality);
    ~TrackFindingProcessor() = default;

    // produce TTTracks
    void produce(const tt::StreamsTrack& inputs,
                 const tt::Streams& inputsAdd,
                 const tt::StreamsStub& stubs,
                 tt::TTTracks& ttTracks,
                 tt::StreamsTrack& outputs);
    // produce StreamsTrack
    void produce(const std::vector<TTTrackRef>& inputs, tt::StreamsTrack& outputs) const;

  private:
    // number of bits used to describe one part of a track (96 bit)
    static constexpr int partial_width = 32;
    // number of track parts arriving per clock tick (1 track per tick)
    static constexpr int partial_in = 3;
    // number of track parts leaving per clock tick (TFP sends 2/3 tracks per clock and link)
    static constexpr int partial_out = 2;
    // type describing one part of a track
    typedef std::bitset<partial_width> PartialFrame;
    // type describing one part of a track together with its edm ref
    typedef std::pair<const TTTrackRef&, PartialFrame> PartialFrameTrack;
    // type representing a track
    struct Track {
      Track(const tt::FrameTrack& frameTrack,
            const tt::Frame& frameTQ,
            const std::vector<TTStubRef>& ttStubRefs,
            const DataFormats* df,
            const trackerTFP::TrackQuality* tq);
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
                 const tt::Streams& inputsAdd,
                 const tt::StreamsStub& stubs,
                 std::vector<std::deque<Track*>>& outputs);
    // emualte data format f/w
    void produce(std::vector<std::deque<Track*>>& inputs, tt::StreamsTrack& outputs) const;
    // produce TTTracks
    void produce(const tt::StreamsTrack& inputs, tt::TTTracks& ouputs) const;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides data formats
    const DataFormats* dataFormats_;
    // provides Track Quality algo and formats
    const trackerTFP::TrackQuality* trackQuality_;
    // storage of tracks
    std::vector<Track> tracks_;
    // b field
    double bfield_;
  };

}  // namespace trklet

#endif
