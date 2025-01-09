#ifndef L1Trigger_TrackerTFP_KalmanFilter_h
#define L1Trigger_TrackerTFP_KalmanFilter_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"
#include "L1Trigger/TrackerTFP/interface/State.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>
#include <utility>

namespace trackerTFP {

  // Class to do helix fit to all tracks in a region.
  class KalmanFilter {
  public:
    typedef State::Stub Stub;
    KalmanFilter(const edm::ParameterSet& iConfig,
                 const tt::Setup* setup,
                 const DataFormats* dataFormats,
                 const LayerEncoding* layerEncoding,
                 KalmanFilterFormats* kalmanFilterFormats,
                 std::vector<TrackKF>& tracks,
                 std::vector<StubKF>& stubs);
    ~KalmanFilter() {}

    // fill output products
    void produce(const std::vector<std::vector<TrackCTB*>>& tracksIn,
                 const std::vector<std::vector<Stub*>>& stubsIn,
                 std::vector<std::vector<TrackKF*>>& tracksOut,
                 std::vector<std::vector<std::vector<StubKF*>>>& stubsOut,
                 int& numAcceptedStates,
                 int& numLostStates,
                 std::deque<std::pair<double, double>>& chi2s);

  private:
    //
    struct Track {
      Track() {}
      Track(int trackId,
            int numConsistent,
            int numConsistentPS,
            double inv2R,
            double phiT,
            double cot,
            double zT,
            double chi20,
            double chi21,
            const TTBV& hitPattern,
            TrackCTB* track,
            const std::vector<StubCTB*>& stubs,
            const std::vector<double>& phi,
            const std::vector<double>& z)
          : trackId_(trackId),
            numConsistent_(numConsistent),
            numConsistentPS_(numConsistentPS),
            inv2R_(inv2R),
            phiT_(phiT),
            cot_(cot),
            zT_(zT),
            chi20_(chi20),
            chi21_(chi21),
            hitPattern_(hitPattern),
            track_(track),
            stubs_(stubs),
            phi_(phi),
            z_(z) {}
      int trackId_;
      int numConsistent_;
      int numConsistentPS_;
      double inv2R_;
      double phiT_;
      double cot_;
      double zT_;
      double chi20_;
      double chi21_;
      TTBV hitPattern_;
      TrackCTB* track_;
      std::vector<StubCTB*> stubs_;
      std::vector<double> phi_;
      std::vector<double> z_;
    };
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // constraints double precision
    double digi(VariableKF var, double val) const { return kalmanFilterFormats_->format(var).digi(val); }
    //
    int integer(VariableKF var, double val) const { return kalmanFilterFormats_->format(var).integer(val); }
    //
    void updateRangeActual(VariableKF var, double val) {
      return kalmanFilterFormats_->format(var).updateRangeActual(val);
    }
    //
    double base(VariableKF var) const { return kalmanFilterFormats_->format(var).base(); }
    //
    int width(VariableKF var) const { return kalmanFilterFormats_->format(var).width(); }
    //
    int inRange(VariableKF var, double val) const { return kalmanFilterFormats_->format(var).inRange(val); }

    // create Proto States
    void createProtoStates(const std::vector<std::vector<TrackCTB*>>& tracksIn,
                           const std::vector<std::vector<Stub*>>& stubsIn,
                           int channel,
                           std::deque<State*>& stream);
    // calulcate seed parameter
    void calcSeeds(std::deque<State*>& stream);
    // apply final cuts
    void finalize(const std::deque<State*>& stream, std::vector<Track>& finals);
    // Transform States into Tracks
    void conv(const std::vector<Track*>& best, std::vector<TrackKF*>& tracks, std::vector<std::vector<StubKF*>>& stubs);
    // adds a layer to states
    void addLayer(std::deque<State*>& stream);
    // adds a layer to states to build seeds
    void addSeedLayer(std::deque<State*>& stream);
    // Assign next combinatoric (i.e. not first in layer) stub to state
    void comb(State*& state);
    // best state selection
    void accumulator(std::vector<Track>& finals, std::vector<Track*>& best);
    // updates state
    void update(State*& state);

    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // provides layer Encoding
    const LayerEncoding* layerEncoding_;
    // provides dataformats of Kalman filter internals
    KalmanFilterFormats* kalmanFilterFormats_;
    // container of output tracks
    std::vector<TrackKF>& tracks_;
    // container of output stubs
    std::vector<StubKF>& stubs_;
    // container of all Kalman Filter states
    std::deque<State> states_;
    //
    std::vector<Track> finals_;
    // current layer used during state propagation
    int layer_;
  };

}  // namespace trackerTFP

#endif
