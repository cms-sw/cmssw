#ifndef L1Trigger_TrackFindingTracklet_KFin_h
#define L1Trigger_TrackFindingTracklet_KFin_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <vector>

namespace trklet {

  /*! \class  trklet::KFin
   *  \brief  Class to emulate transformation of tracklet tracks and stubs into TMTT format
   *  \author Thomas Schuh
   *  \date   2021, Dec
   */
  class KFin {
  public:
    KFin(const edm::ParameterSet& iConfig,
         const tt::Setup* setup_,
         const trackerTFP::DataFormats* dataFormats,
         const trackerTFP::LayerEncoding* layerEncoding,
         const ChannelAssignment* channelAssignment,
         const Settings* settings,
         int region);
    ~KFin() {}

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
      Stub(const TTStubRef& ttStubRef, int layer, double r, double phi, double z, bool psTilt)
          : valid_(true), ttStubRef_(ttStubRef), layer_(layer), r_(r), phi_(phi), z_(z), psTilt_(psTilt) {}
      bool valid_;
      TTStubRef ttStubRef_;
      int layer_;
      // radius w.r.t. chosenRofPhi in cm
      double r_;
      // phi residual in rad
      double phi_;
      // z residual in cm
      double z_;
      // phi uncertainty * sqrt(12) + additional terms in rad
      double dPhi_;
      // z uncertainty * sqrt(12) + additional terms in cm
      double dZ_;
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
