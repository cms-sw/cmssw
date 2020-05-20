#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"

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

void SLHCEvent::addL1SimTrack(
    int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz) {
  vx -= x_offset_;
  vy -= y_offset_;
  L1SimTrack simtrack(eventid, trackid, type, pt, eta, phi, vx, vy, vz);
  simtracks_.push_back(simtrack);
}

bool SLHCEvent::addStub(int layer,
                        int ladder,
                        int module,
                        int strip,
                        int eventid,
                        vector<int> tps,
                        double pt,
                        double bend,
                        double x,
                        double y,
                        double z,
                        int isPSmodule,
                        int isFlipped) {
  if (layer > 999 && layer < 1999 && z < 0.0) {
    layer += 1000;
  }

  layer--;
  x -= x_offset_;
  y -= y_offset_;

  L1TStub stub(
      eventid, tps, -1, -1, layer, ladder, module, strip, x, y, z, -1.0, -1.0, pt, bend, isPSmodule, isFlipped);

  stubs_.push_back(stub);
  return true;
}

SLHCEvent::SLHCEvent(istream& in) {
  string tmp;
  in >> tmp;
  while (tmp == "Map:") {
    in >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp;
    in >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp;
  }
  if (tmp == "EndMap") {
    in >> tmp;
  }
  if (tmp != "Event:") {
    edm::LogVerbatim("Tracklet") << "Expected to read 'Event:' but found:" << tmp;
    if (tmp.empty()) {
      edm::LogVerbatim("Tracklet") << "WARNING: fewer events to process than specified!";
      return;
    } else {
      edm::LogVerbatim("Tracklet") << "ERROR, aborting reading file";
      abort();
    }
  }
  in >> eventnum_;

  // read the SimTracks
  in >> tmp;
  while (tmp != "SimTrackEnd") {
    if (!(tmp == "SimTrack:" || tmp == "SimTrackEnd")) {
      edm::LogVerbatim("Tracklet") << "Expected to read 'SimTrack:' or 'SimTrackEnd' but found:" << tmp;
      abort();
    }
    int eventid;
    int trackid;
    int type;
    string pt_str;
    string eta_str;
    string phi_str;
    string vx_str;
    string vy_str;
    string vz_str;
    double pt;
    double eta;
    double phi;
    double vx;
    double vy;
    double vz;
    in >> eventid >> trackid >> type >> pt_str >> eta_str >> phi_str >> vx_str >> vy_str >> vz_str;
    pt = strtod(pt_str.c_str(), nullptr);
    eta = strtod(eta_str.c_str(), nullptr);
    phi = strtod(phi_str.c_str(), nullptr);
    vx = strtod(vx_str.c_str(), nullptr);
    vy = strtod(vy_str.c_str(), nullptr);
    vz = strtod(vz_str.c_str(), nullptr);
    vx -= x_offset_;
    vy -= y_offset_;
    L1SimTrack simtrack(eventid, trackid, type, pt, eta, phi, vx, vy, vz);
    simtracks_.push_back(simtrack);
    in >> tmp;
  }

  int nlayer[11];
  for (int i = 0; i < 10; i++) {
    nlayer[i] = 0;
  }

  int oldlayer = 0;
  int oldladder = 0;
  int oldmodule = 0;
  int oldcbc = -1;
  int count = 1;
  double oldz = -1000.0;

  //read stubs
  in >> tmp;
  while (tmp != "StubEnd") {
    if (!in.good()) {
      edm::LogVerbatim("Tracklet") << "File not good";
      abort();
    };
    if (!(tmp == "Stub:" || tmp == "StubEnd")) {
      edm::LogVerbatim("Tracklet") << "Expected to read 'Stub:' or 'StubEnd' but found:" << tmp;
      abort();
    }
    int layer;
    int ladder;
    int module;
    int eventid;
    vector<int> tps;
    int strip;
    double pt;
    double x;
    double y;
    double z;
    double bend;
    int isPSmodule;
    int isFlipped;

    unsigned int ntps;

    in >> layer >> ladder >> module >> strip >> eventid >> pt >> x >> y >> z >> bend >> isPSmodule >> isFlipped >> ntps;

    for (unsigned int itps = 0; itps < ntps; itps++) {
      int tp;
      in >> tp;
      tps.push_back(tp);
    }

    if (layer > 999 && layer < 1999 && z < 0.0) {  //negative disk
      layer += 1000;
    }

    int cbc = strip / 126;
    if (layer > 3 && layer == oldlayer && ladder == oldladder && module == oldmodule && cbc == oldcbc &&
        std::abs(oldz - z) < 1.0) {
      count++;
    } else {
      oldlayer = layer;
      oldladder = ladder;
      oldmodule = module;
      oldcbc = cbc;
      oldz = z;
      count = 1;
    }

    layer--;
    x -= x_offset_;
    y -= y_offset_;

    if (layer < 10)
      nlayer[layer]++;

    L1TStub stub(
        eventid, tps, -1, -1, layer, ladder, module, strip, x, y, z, -1.0, -1.0, pt, bend, isPSmodule, isFlipped);

    in >> tmp;

    double t = std::abs(stub.z()) / stub.r();
    double eta = asinh(t);

    if (std::abs(eta) < 2.6 && count <= 100) {
      stubs_.push_back(stub);
    }
  }
}

void SLHCEvent::write(ofstream& out) {
  out << "Event: " << eventnum_ << endl;

  for (auto& simtrack : simtracks_) {
    simtrack.write(out);
  }
  out << "SimTrackEnd" << endl;

  for (auto& stub : stubs_) {
    stub.write(out);
  }
  out << "StubEnd" << endl;
}

void SLHCEvent::write(ostream& out) {
  out << "Event: " << eventnum_ << endl;

  for (auto& simtrack : simtracks_) {
    simtrack.write(out);
  }
  out << "SimTrackEnd" << endl;

  for (auto& stub : stubs_) {
    stub.write(out);
  }
  out << "StubEnd" << endl;
}

unsigned int SLHCEvent::layersHit(int tpid, int& nlayers, int& ndisks) {
  int l1 = 0;
  int l2 = 0;
  int l3 = 0;
  int l4 = 0;
  int l5 = 0;
  int l6 = 0;

  int d1 = 0;
  int d2 = 0;
  int d3 = 0;
  int d4 = 0;
  int d5 = 0;

  for (auto& stub : stubs_) {
    if (stub.tpmatch(tpid)) {
      if (stub.layer() == 0)
        l1 = 1;
      if (stub.layer() == 1)
        l2 = 1;
      if (stub.layer() == 2)
        l3 = 1;
      if (stub.layer() == 3)
        l4 = 1;
      if (stub.layer() == 4)
        l5 = 1;
      if (stub.layer() == 5)
        l6 = 1;

      if (abs(stub.disk()) == 1)
        d1 = 1;
      if (abs(stub.disk()) == 2)
        d2 = 1;
      if (abs(stub.disk()) == 3)
        d3 = 1;
      if (abs(stub.disk()) == 4)
        d4 = 1;
      if (abs(stub.disk()) == 5)
        d5 = 1;
    }
  }

  nlayers = l1 + l2 + l3 + l4 + l5 + l6;
  ndisks = d1 + d2 + d3 + d4 + d5;

  return l1 + 2 * l2 + 4 * l3 + 8 * l4 + 16 * l5 + 32 * l6 + 64 * d1 + 128 * d2 + 256 * d3 + 512 * d4 + 1024 * d5;
}

int SLHCEvent::getSimtrackFromSimtrackid(int simtrackid, int eventid) const {
  for (unsigned int i = 0; i < simtracks_.size(); i++) {
    if (simtracks_[i].trackid() == simtrackid && simtracks_[i].eventid() == eventid)
      return i;
  }
  return -1;
}
