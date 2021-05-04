#include "L1Trigger/TrackFindingTracklet/interface/TrackDer.h"

using namespace std;
using namespace trklet;

TrackDer::TrackDer() {
  for (unsigned int i = 0; i < N_FITSTUB; i++) {
    irinvdphi_[i] = 9999999;
    irinvdzordr_[i] = 9999999;
    iphi0dphi_[i] = 9999999;
    iphi0dzordr_[i] = 9999999;
    itdphi_[i] = 9999999;
    itdzordr_[i] = 9999999;
    iz0dphi_[i] = 9999999;
    iz0dzordr_[i] = 9999999;

    rinvdphi_[i] = 0.0;
    rinvdzordr_[i] = 0.0;
    phi0dphi_[i] = 0.0;
    phi0dzordr_[i] = 0.0;
    tdphi_[i] = 0.0;
    tdzordr_[i] = 0.0;
    z0dphi_[i] = 0.0;
    z0dzordr_[i] = 0.0;
  }

  for (unsigned int i = 0; i < N_PSLAYER; i++) {
    for (unsigned int j = 0; j < N_PSLAYER; j++) {
      tdzcorr_[i][j] = 0.0;
      z0dzcorr_[i][j] = 0.0;
    }
  }
}

void TrackDer::setIndex(int layermask, int diskmask, int alphamask, int irinv) {
  layermask_ = layermask;
  diskmask_ = diskmask;
  alphamask_ = alphamask;
  irinv_ = irinv;
}

void TrackDer::fill(int t, double MinvDt[N_FITPARAM][N_FITSTUB * 2], int iMinvDt[N_FITPARAM][N_FITSTUB * 2]) const {
  unsigned int nlayer = 0;
  if (layermask_ & 1)
    nlayer++;
  if (layermask_ & 2)
    nlayer++;
  if (layermask_ & 4)
    nlayer++;
  if (layermask_ & 8)
    nlayer++;
  if (layermask_ & 16)
    nlayer++;
  if (layermask_ & 32)
    nlayer++;
  int sign = 1;
  if (t < 0)
    sign = -1;
  for (unsigned int i = 0; i < N_FITSTUB; i++) {
    MinvDt[0][2 * i] = rinvdphi_[i];
    MinvDt[1][2 * i] = phi0dphi_[i];
    MinvDt[2][2 * i] = sign * tdphi_[i];
    MinvDt[3][2 * i] = sign * z0dphi_[i];
    MinvDt[0][2 * i + 1] = rinvdzordr_[i];
    MinvDt[1][2 * i + 1] = phi0dzordr_[i];
    MinvDt[2][2 * i + 1] = tdzordr_[i];
    MinvDt[3][2 * i + 1] = z0dzordr_[i];
    iMinvDt[0][2 * i] = irinvdphi_[i];
    iMinvDt[1][2 * i] = iphi0dphi_[i];
    iMinvDt[2][2 * i] = sign * itdphi_[i];
    iMinvDt[3][2 * i] = sign * iz0dphi_[i];
    iMinvDt[0][2 * i + 1] = irinvdzordr_[i];
    iMinvDt[1][2 * i + 1] = iphi0dzordr_[i];
    iMinvDt[2][2 * i + 1] = itdzordr_[i];
    iMinvDt[3][2 * i + 1] = iz0dzordr_[i];
    if (i < nlayer) {
      MinvDt[0][2 * i + 1] *= sign;
      MinvDt[1][2 * i + 1] *= sign;
      iMinvDt[0][2 * i + 1] *= sign;
      iMinvDt[1][2 * i + 1] *= sign;
    } else {
      MinvDt[2][2 * i + 1] *= sign;
      MinvDt[3][2 * i + 1] *= sign;
      iMinvDt[2][2 * i + 1] *= sign;
      iMinvDt[3][2 * i + 1] *= sign;
    }
  }
}
