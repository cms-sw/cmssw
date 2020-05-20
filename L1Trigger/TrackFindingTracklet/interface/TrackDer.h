//
// This class holdes the 'deriviatives' used in the linearized chi^2 fit.
// This is also referred to as the weight matrix which is used to weight
// the residuls when calculating the updated track parameters.
//
//
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackDer_h
#define L1Trigger_TrackFindingTracklet_interface_TrackDer_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

namespace trklet {

  class TrackDer {
  public:
    TrackDer();

    ~TrackDer() = default;

    void setIndex(int layermask, int diskmask, int alphamask, int irinv);

    int layerMask() const { return layermask_; }
    int diskMask() const { return diskmask_; }
    int alphaMask() const { return alphamask_; }
    int irinv() const { return irinv_; }

    void setirinvdphi(int i, int irinvdphi) { irinvdphi_[i] = irinvdphi; }
    void setirinvdzordr(int i, int irinvdzordr) { irinvdzordr_[i] = irinvdzordr; }
    void setiphi0dphi(int i, int iphi0dphi) { iphi0dphi_[i] = iphi0dphi; }
    void setiphi0dzordr(int i, int iphi0dzordr) { iphi0dzordr_[i] = iphi0dzordr; }
    void setitdphi(int i, int itdphi) { itdphi_[i] = itdphi; }
    void setitdzordr(int i, int itdzordr) { itdzordr_[i] = itdzordr; }
    void setiz0dphi(int i, int iz0dphi) { iz0dphi_[i] = iz0dphi; }
    void setiz0dzordr(int i, int iz0dzordr) { iz0dzordr_[i] = iz0dzordr; }

    void setitdzcorr(int i, int j, int itdzcorr) { itdzcorr_[i][j] = itdzcorr; }
    void setiz0dzcorr(int i, int j, int iz0dzcorr) { iz0dzcorr_[i][j] = iz0dzcorr; }

    void setrinvdphi(int i, double rinvdphi) { rinvdphi_[i] = rinvdphi; }
    void setrinvdzordr(int i, double rinvdzordr) { rinvdzordr_[i] = rinvdzordr; }
    void setphi0dphi(int i, double phi0dphi) { phi0dphi_[i] = phi0dphi; }
    void setphi0dzordr(int i, double phi0dzordr) { phi0dzordr_[i] = phi0dzordr; }
    void settdphi(int i, double tdphi) { tdphi_[i] = tdphi; }
    void settdzordr(int i, double tdzordr) { tdzordr_[i] = tdzordr; }
    void setz0dphi(int i, double z0dphi) { z0dphi_[i] = z0dphi; }
    void setz0dzordr(int i, double z0dzordr) { z0dzordr_[i] = z0dzordr; }

    void settdzcorr(int i, int j, double tdzcorr) { tdzcorr_[i][j] = tdzcorr; }
    void setz0dzcorr(int i, int j, double z0dzcorr) { z0dzcorr_[i][j] = z0dzcorr; }

    double rinvdphi(int i) const { return rinvdphi_[i]; }
    double rinvdzordr(int i) const { return rinvdzordr_[i]; }
    double phi0dphi(int i) const { return phi0dphi_[i]; }
    double phi0dzordr(int i) const { return phi0dzordr_[i]; }
    double tdphi(int i) const { return tdphi_[i]; }
    double tdzordr(int i) const { return tdzordr_[i]; }
    double z0dphi(int i) const { return z0dphi_[i]; }
    double z0dzordr(int i) const { return z0dzordr_[i]; }

    double tdzcorr(int i, int j) const { return tdzcorr_[i][j]; }
    double z0dzcorr(int i, int j) const { return z0dzcorr_[i][j]; }

    double irinvdphi(int i) const { return irinvdphi_[i]; }
    double irinvdzordr(int i) const { return irinvdzordr_[i]; }
    double iphi0dphi(int i) const { return iphi0dphi_[i]; }
    double iphi0dzordr(int i) const { return iphi0dzordr_[i]; }
    double itdphi(int i) const { return itdphi_[i]; }
    double itdzordr(int i) const { return itdzordr_[i]; }
    double iz0dphi(int i) const { return iz0dphi_[i]; }
    double iz0dzordr(int i) const { return iz0dzordr_[i]; }

    int itdzcorr(int i, int j) const { return itdzcorr_[i][j]; }
    int iz0dzcorr(int i, int j) const { return iz0dzcorr_[i][j]; }

    void settpar(double t) { t_ = t; }
    double tpar() const { return t_; }

    void fill(int t, double MinvDt[4][12], int iMinvDt[4][12]) const;

  private:
    int irinvdphi_[N_FITSTUB];
    int irinvdzordr_[N_FITSTUB];
    int iphi0dphi_[N_FITSTUB];
    int iphi0dzordr_[N_FITSTUB];
    int itdphi_[N_FITSTUB];
    int itdzordr_[N_FITSTUB];
    int iz0dphi_[N_FITSTUB];
    int iz0dzordr_[N_FITSTUB];

    int itdzcorr_[N_PSLAYER][N_PSLAYER];
    int iz0dzcorr_[N_PSLAYER][N_PSLAYER];

    double rinvdphi_[N_FITSTUB];
    double rinvdzordr_[N_FITSTUB];
    double phi0dphi_[N_FITSTUB];
    double phi0dzordr_[N_FITSTUB];
    double tdphi_[N_FITSTUB];
    double tdzordr_[N_FITSTUB];
    double z0dphi_[N_FITSTUB];
    double z0dzordr_[N_FITSTUB];

    double tdzcorr_[N_PSLAYER][N_PSLAYER];
    double z0dzcorr_[N_PSLAYER][N_PSLAYER];

    double t_;

    int layermask_;
    int diskmask_;
    int alphamask_;
    int irinv_;
  };

};  // namespace trklet
#endif
