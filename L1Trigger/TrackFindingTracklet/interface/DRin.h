#ifndef L1Trigger_TrackFindingTracklet_DRin_h
#define L1Trigger_TrackFindingTracklet_DRin_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <vector>
#include <deque>

namespace trklet {

  /*! \class  trklet::DRin
   *  \brief  Class to emulate transformation of tracklet tracks and stubs into TMTT format
   *          and routing of seed type streams into inv2R streams
   *  \author Thomas Schuh
   *  \date   2023, Jan
   */
  class DRin {
  public:
    DRin(const edm::ParameterSet& iConfig,
         const tt::Setup* setup_,
         const trackerTFP::DataFormats* dataFormats,
         const trackerTFP::LayerEncoding* layerEncoding,
         const ChannelAssignment* channelAssignment,
         const Settings* settings,
         int region);
    ~DRin() {}
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub);
    // fill output products
    void produce(tt::StreamsStub& accpetedStubs,
                 tt::StreamsTrack& acceptedTracks,
                 tt::StreamsStub& lostStubs,
                 tt::StreamsTrack& lostTracks);

  private:
    // truncates double precision of val into base precision, +1.e-12 restores robustness of addition of 2 digitised values
    double digi(double val, double base) const { return (floor(val / base + 1.e-12) + .5) * base; }
    // basetransformation of val from baseLow into baseHigh using widthMultiplier bit multiplication
    double redigi(double val, double baseLow, double baseHigh, int widthMultiplier) const;
    struct Stub {
      Stub(const TTStubRef& ttStubRef,
           int layer,
           int layerDet,
           bool seed,
           int stubId,
           double r,
           double phi,
           double z,
           bool psTilt)
          : valid_(true),
            ttStubRef_(ttStubRef),
            layer_(layer),
            layerDet_(layerDet),
            layerKF_(-1),
            seed_(seed),
            stubId_(stubId),
            r_(r),
            phi_(phi),
            z_(z),
            psTilt_(psTilt) {}
      bool valid_;
      TTStubRef ttStubRef_;
      // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
      int layer_;
      // layer id [0-5] barrel [6-10] end cap discs
      int layerDet_;
      // layer id [0-6] counted from inside-out along track
      int layerKF_;
      // true if stub was part of the seed
      bool seed_;
      // traclet stub id
      int stubId_;
      // radius w.r.t. chosenRofPhi in cm
      double r_;
      // phi residual in rad
      double phi_;
      // z residual in cm
      double z_;
      // true if barrel tilted module or encap PS module
      bool psTilt_;
    };
    struct Track {
      static constexpr int max_ = 8;
      Track() { stubs_.reserve(max_); }
      Track(const TTTrackRef& ttTrackRef,
            bool valid,
            double inv2R,
            double phiT,
            double cot,
            double zT,
            const std::vector<Stub*>& stubs)
          : ttTrackRef_(ttTrackRef),
            valid_(valid),
            sector_(-1),
            inv2R_(inv2R),
            phiT_(phiT),
            cot_(cot),
            zT_(zT),
            stubs_(stubs) {}
      TTTrackRef ttTrackRef_;
      bool valid_;
      TTBV maybe_;
      int sector_;
      double inv2R_;
      double phiT_;
      double cot_;
      double zT_;
      std::vector<Stub*> stubs_;
    };
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // true if truncation is enbaled
    bool enableTruncation_;
    // stub residuals are recalculated from seed parameter and TTStub position
    bool useTTStubResiduals_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const trackerTFP::DataFormats* dataFormats_;
    // helper class to encode layer
    const trackerTFP::LayerEncoding* layerEncoding_;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_;
    // provides tracklet constants
    const Settings* settings_;
    // processing region (0 - 8) aka processing phi nonant
    const int region_;
    // storage of input tracks
    std::vector<Track> tracks_;
    // storage of input stubs
    std::vector<Stub> stubs_;
    // h/w liked organized pointer to input tracks
    std::vector<std::vector<Track*>> input_;
    // unified tracklet digitisation granularity
    double baseUinv2R_;
    double baseUphiT_;
    double baseUcot_;
    double baseUzT_;
    double baseUr_;
    double baseUphi_;
    double baseUz_;
    // KF input format digitisation granularity (identical to TMTT)
    double baseLinv2R_;
    double baseLphiT_;
    double baseLcot_;
    double baseLzT_;
    double baseLr_;
    double baseLphi_;
    double baseLz_;
    // Finer granularity (by powers of 2) than the TMTT one. Used to transform from Tracklet to TMTT base.
    double baseHinv2R_;
    double baseHphiT_;
    double baseHcot_;
    double baseHzT_;
    double baseHr_;
    double baseHphi_;
    double baseHz_;
    // digitisation granularity used for inverted cot(theta)
    double baseInvCot_;
  };

}  // namespace trklet

#endif
