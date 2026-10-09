#ifndef L1Trigger_TrackFindingTracklet_DuplicateRemoval_h
#define L1Trigger_TrackFindingTracklet_DuplicateRemoval_h

#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"

#include <vector>

namespace trklet {

  /*! \class  trklet::DuplicateRemoval
   *  \brief  Class to bit- and clock-accurate emulate duplicate removal
   *          DR identifies duplicates based on pairs of tracks that share stubs in at least 3 layers.
   *          It keeps the first such track in each pair. The Track order is determined by TrackMultiplexer,
   *          provided by ProducerTM.
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class DuplicateRemoval {
  public:
    DuplicateRemoval(const Setup*, const DataFormats*, int, const TTDTC&);
    ~DuplicateRemoval() = default;
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack&, const tt::StreamsStub&);
    // fill output products
    void produce(tt::StreamsTrack&, tt::StreamsStub&);

  private:
    // remove duplicated tracks, no merge of stubs, one stub per layer expected
    void algo();
    // unify stubs
    void unify();
    // calc stub position
    void pos();
    // replace stubs with DTC stubs
    void dtc();
    // replace stubs with TT stubs
    void tt();
    // calc stub uncertainties
    void delta();
    // base transformation
    void redigi();
    // store output
    void store(tt::StreamsTrack&, tt::StreamsStub&) const;
    struct Stub {
      Stub(const TTStubRef& ttStubRef) : ttStubRef_(ttStubRef), r_(0.), phi_(0.), z_(0.) {}
      TTStubRef ttStubRef_;
      const trackerDTC::SensorModule* sm_;
      int layer_;
      int stubId_;
      double r_;
      double phi_;
      double z_;
      double dPhi_;
      double dZ_;
    };
    struct Track {
      TrackDR trackDR_;
      double inv2R_;
      double phi0_;
      double cot_;
      double z0_;
      std::vector<Stub*> stubs_;
    };
    // compares two tracks, returns true if those are considered duplicates
    bool equalEnough(Track* t0, Track* t1) const;
    // provides run-time constants
    const Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // processing region (0 - 8) aka processing phi nonant
    const int region_;
    // storage of input tracks
    std::vector<Track> tracks_;
    // storage of input stubs
    std::vector<Stub> stubs_;
    // h/w liked organized pointer to input tracks
    std::vector<Track*> stream_;
    // DTC stubs
    std::vector<tt::FrameStub> dtc_;
    // internal used bases
    double baseR_;
    double basePhi_;
    double baseZ_;
  };

}  // namespace trklet

#endif
