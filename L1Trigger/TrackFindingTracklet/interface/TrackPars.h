#ifndef L1Trigger_TrackFindingTracklet_interface_TrackPars_h
#define L1Trigger_TrackFindingTracklet_interface_TrackPars_h

namespace trklet {

  template <class T>
  class TrackPars {
  public:
    TrackPars() = default;

    TrackPars(T rinv, T phi0, T d0, T t, T z0) {
      rinv_ = rinv;
      phi0_ = phi0;
      d0_ = d0;
      t_ = t;
      z0_ = z0;
    }

    ~TrackPars() = default;

    void init(T rinv, T phi0, T d0, T t, T z0) {
      rinv_ = rinv;
      phi0_ = phi0;
      d0_ = d0;
      t_ = t;
      z0_ = z0;
    }

    const T& rinv() const { return rinv_; }
    const T& phi0() const { return phi0_; }
    const T& d0() const { return d0_; }
    const T& t() const { return t_; }
    const T& z0() const { return z0_; }

    T& rinv() { return rinv_; }
    T& phi0() { return phi0_; }
    T& d0() { return d0_; }
    T& t() { return t_; }
    T& z0() { return z0_; }

  private:
    T rinv_;
    T phi0_;
    T d0_;
    T t_;
    T z0_;
  };

};  // namespace trklet
#endif
