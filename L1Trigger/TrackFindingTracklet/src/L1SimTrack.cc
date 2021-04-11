#include "L1Trigger/TrackFindingTracklet/interface/L1SimTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

L1SimTrack::L1SimTrack() {
  eventid_ = -1;
  trackid_ = -1;
}

L1SimTrack::L1SimTrack(
    int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz) {
  eventid_ = eventid;
  trackid_ = trackid;
  type_ = type;
  pt_ = pt;
  eta_ = eta;
  phi_ = phi;
  vx_ = vx;
  vy_ = vy;
  vz_ = vz;
}

void L1SimTrack::write(ofstream& out) {
  if (pt_ > -2.0) {
    out << "SimTrack: " << eventid_ << "\t" << trackid_ << "\t" << type_ << "\t" << pt_ << "\t" << eta_ << "\t" << phi_
        << "\t" << vx_ << "\t" << vy_ << "\t" << vz_ << "\t" << endl;
  }
}

void L1SimTrack::write(ostream& out) {
  if (pt_ > -2) {
    out << "SimTrack: " << eventid_ << "\t" << trackid_ << "\t" << type_ << "\t" << pt_ << "\t" << eta_ << "\t" << phi_
        << "\t" << vx_ << "\t" << vy_ << "\t" << vz_ << "\t" << endl;
  }
}
