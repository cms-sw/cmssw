#ifndef L1Trigger_TrackFindingTracklet_KalmanFilter_h
#define L1Trigger_TrackFindingTracklet_KalmanFilter_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilterFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/State.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"

#include <vector>
#include <deque>

namespace trklet {

  /*! \class  trklet::KalmanFilter
   *  \brief  Class to do helix fit to all tracks in a region.
   *          All variable names & equations come from Fruhwirth KF paper
   *          http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
   *          Summary of variables:
   *          m = hit position (phi,z)
   *          V = hit position 2x2 covariance matrix in (phi,z).
   *          x = helix params
   *          C = helix params 4x4 covariance matrix
   *          r = residuals
   *          H = 2x4 derivative matrix (expected stub position w.r.t. helix params)
   *          K = KF gain 2x2 matrix
   *          x' & C': Updated values of x & C after KF iteration
   *          Boring: F = unit matrix; pxcov = C
   *          Summary of equations:
   *          S = H*C (2x4 matrix); St = Transpose S
   *          R = V + H*C*Ht (KF paper) = V + H*St (used here at simpler): 2x2 matrix
   *          Rinv = Inverse R
   *          K = St * Rinv : 2x2 Kalman gain matrix * det(R)
   *          r = m - H*x
   *          x' = x + K*r
   *          C' = C - K*H*C (KF paper) = C - K*S (used here as simpler)
   *  \author Thomas Schuh
   *  \date   2024, Sep
   */
  class KalmanFilter {
  public:
    typedef State::Stub Stub;
    KalmanFilter(const tt::Setup* setup,
                 const DataFormats* dataFormats,
                 KalmanFilterFormats* kalmanFilterFormats,
                 tmtt::Settings* settings,
                 tmtt::KFParamsComb* tmtt,
                 int region,
                 tt::TTTracks& ttTracks);
    ~KalmanFilter() {}
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub);
    // fill output products
    void produce(tt::StreamsStub& streamsStub,
                 tt::StreamsTrack& streamsTrack,
                 int& numAcceptedStates,
                 int& numLostStates);

  private:
    //
    struct Track {
      Track() {}
      Track(int trackId,
            int numConsistent,
            int numConsistentPS,
            double d0,
            const TTBV& hitPattern,
            const TrackKF& trackKF,
            const std::vector<StubKF>& stubsKF)
          : trackId_(trackId),
            numConsistent_(numConsistent),
            numConsistentPS_(numConsistentPS),
            d0_(d0),
            hitPattern_(hitPattern),
            trackKF_(trackKF),
            stubsKF_(stubsKF) {}
      int trackId_;
      int numConsistent_;
      int numConsistentPS_;
      double d0_;
      TTBV hitPattern_;
      TrackKF trackKF_;
      std::vector<StubKF> stubsKF_;
    };
    // call old KF
    void simulate(tt::StreamsStub& streamsStub, tt::StreamsTrack& streamsTrack);
    // constraints double precision
    double digi(VariableKF var, double val) { return kalmanFilterFormats_->format(var).digi(val); }
    //
    int integer(VariableKF var, double val) { return kalmanFilterFormats_->format(var).integer(val); }
    //
    void updateRangeActual(VariableKF var, double val) {
      return kalmanFilterFormats_->format(var).updateRangeActual(val);
    }
    //
    double base(VariableKF var) { return kalmanFilterFormats_->format(var).base(); }
    //
    int width(VariableKF var) { return kalmanFilterFormats_->format(var).width(); }
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // calculates the helix params & their cov. matrix from a pair of stubs
    void calcSeeds();
    // Transform States into output products
    void conv(tt::StreamsStub& streamsStub, tt::StreamsTrack& streamsTrack);
    // adds a layer to states
    void addLayer();
    // adds a layer to states to build seeds
    void addSeedLayer();
    // Assign next combinatoric (i.e. not first in layer) stub to state
    void comb(State*& state);
    // apply final cuts
    void finalize();
    // best state selection
    void accumulator();
    // updates state
    void update(State*& state) { setup_->kfUse5ParameterFit() ? update5(state) : update4(state); }
    // updates state using 4 paramter fit
    void update4(State*& state);
    // updates state using 5 parameter fit
    void update5(State*& state);

    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // provides dataformats of Kalman filter internals
    KalmanFilterFormats* kalmanFilterFormats_;
    //
    tmtt::Settings* settings_;
    //
    tmtt::KFParamsComb* tmtt_;
    // processing region
    int region_;
    //
    tt::TTTracks& ttTracks_;
    // container of tracks
    std::vector<TrackDR> tracks_;
    // container of stubs
    std::vector<Stub> stubs_;
    // container of all Kalman Filter states
    std::deque<State> states_;
    // processing stream
    std::deque<State*> stream_;
    //
    std::vector<Track> finals_;
    // current layer used during state propagation
    int layer_;
    //
    std::vector<double> zTs_;
  };

}  // namespace trklet

#endif