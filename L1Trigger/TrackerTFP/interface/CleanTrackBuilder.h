#ifndef L1Trigger_TrackerTFP_CleanTrackBuilder_h
#define L1Trigger_TrackerTFP_CleanTrackBuilder_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to clean and transform stream of stubs into a stream of tracks with one stub stream per kf layer
  class CleanTrackBuilder {
  public:
    CleanTrackBuilder(const tt::Setup* setup,
                      const DataFormats* dataFormats,
                      const LayerEncoding* layerEncoding,
                      const DataFormat& cot,
                      std::vector<StubCTB>& stubs,
                      std::vector<TrackCTB>& tracks);
    ~CleanTrackBuilder() {}
    // fill output products
    void produce(const std::vector<std::vector<StubHT*>>& streamsIn,
                 std::vector<std::deque<TrackCTB*>>& regionTracks,
                 std::vector<std::vector<std::deque<StubCTB*>>>& regionStubs);
    void put(TrackCTB* track, const std::vector<std::vector<StubCTB*>>& stubs, int region, tt::TTTracks& ttTracks) const;

  private:
    // struct to represent internal stubs
    struct Stub {
      // construct Stub from StubHT
      Stub(StubHT* stub, int trackId, const TTBV& hitsPhi, const TTBV& hitsZ, int layerId, double dPhi, double dZ)
          : stubHT_(stub),
            trackId_(trackId),
            hitsPhi_(hitsPhi),
            hitsZ_(hitsZ),
            layerId_(layerId),
            dPhi_(dPhi),
            dZ_(dZ) {}
      //
      void update(const TTBV& phi, const TTBV& z, std::vector<int>& ids, int max);
      // original ht stub
      StubHT* stubHT_;
      //
      bool valid_ = true;
      //
      int trackId_;
      //
      int stubId_ = -1;
      //
      TTBV hitsPhi_;
      //
      TTBV hitsZ_;
      //
      int layerId_;
      //
      double dPhi_;
      //
      double dZ_;
    };

    // struct to represent internal tracks
    struct Track {
      // construct Track from Stubs
      Track(const tt::Setup* setup,
            int trackId,
            const TTBV& hitsPhi,
            const TTBV& hitsZ,
            const std::vector<Stub*>& stubs,
            double inv2R);
      //
      bool valid_;
      // stubs
      std::vector<Stub*> stubs_;
      // track id
      int trackId_;
      //
      TTBV hitsPhi_;
      //
      TTBV hitsZ_;
      //
      double inv2R_;
      // size: number of stubs on most occupied layer
      int size_;
    };
    //
    void cleanStream(const std::vector<StubHT*>& input,
                     std::deque<Track*>& tracks,
                     std::deque<Stub*>& stubs,
                     int channelId);
    // run single track through r-phi and r-z hough transform
    void cleanTrack(const std::vector<StubHT*>& track,
                    std::deque<Track*>& tracks,
                    std::deque<Stub*>& stubs,
                    double inv2R,
                    int zT,
                    int trackId);
    //
    void route(std::vector<std::deque<Track*>>& inputs, std::deque<Track*>& output) const;
    //
    void route(std::vector<std::deque<Stub*>>& input, std::vector<std::deque<Stub*>>& outputs) const;
    //
    void sort(std::deque<Track*>& track, std::vector<std::deque<Stub*>>& stubs) const;
    //
    void convert(const std::deque<Track*>& iTracks,
                 const std::vector<std::deque<Stub*>>& iStubs,
                 std::deque<TrackCTB*>& oTracks,
                 std::vector<std::deque<StubCTB*>>& oStubs);
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    //
    const LayerEncoding* layerEncoding_;
    //
    DataFormat cot_;
    // container of internal stubs
    std::vector<Stub> stubs_;
    // container of internal tracks
    std::vector<Track> tracks_;
    // container of output stubs
    std::vector<StubCTB>& stubsCTB_;
    // container of output tracks
    std::vector<TrackCTB>& tracksCTB_;
    // number of output channel
    int numChannelOut_;
    // number of channel
    int numChannel_;
    // number of processing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
    int wlayer_;
    const DataFormat& r_;
    const DataFormat& phi_;
    const DataFormat& z_;
    const DataFormat& phiT_;
    const DataFormat& zT_;
    int numBinsInv2R_;
    int numBinsPhiT_;
    int numBinsCot_;
    int numBinsZT_;
    double baseInv2R_;
    double basePhiT_;
    double baseCot_;
    double baseZT_;
  };

}  // namespace trackerTFP

#endif