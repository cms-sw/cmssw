#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

void SLHCEvent::addL1SimTrack(
    int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz) {
  L1SimTrack simtrack(eventid, trackid, type, pt, eta, phi, vx, vy, vz);
  simtracks_.push_back(simtrack);
}

bool SLHCEvent::addStub(string DTClink,
                        int region,
                        int layerdisk,
                        string stubword,
                        int isPSmodule,
                        int isFlipped,
                        double x,
                        double y,
                        double z,
                        double bend,
                        double strip,
                        vector<int> tps) {
  L1TStub stub(DTClink, region, layerdisk, stubword, isPSmodule, isFlipped, x, y, z, bend, strip, tps);

  stubs_.push_back(stub);
  return true;
}

SLHCEvent::SLHCEvent(istream& in) {
  string tmp;
  in >> tmp;
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
    double pt;
    double eta;
    double phi;
    double vx;
    double vy;
    double vz;
    in >> eventid >> trackid >> type >> pt >> eta >> phi >> vx >> vy >> vz;
    L1SimTrack simtrack(eventid, trackid, type, pt, eta, phi, vx, vy, vz);
    simtracks_.push_back(simtrack);
    in >> tmp;
  }

  //read stubs
  in >> tmp;
  while (tmp != "Stubend") {
    if (!in.good()) {
      edm::LogVerbatim("Tracklet") << "File not good (SLHCEvent)";
      abort();
    };
    if (!(tmp == "Stub:" || tmp == "Stubend")) {
      edm::LogVerbatim("Tracklet") << "Expected to read 'Stub:' or 'StubEnd' but found:" << tmp;
      abort();
    }
    string DTClink;
    int region;
    int layerdisk;
    string stubword;
    int isPSmodule;
    int isFlipped;
    double x;
    double y;
    double z;
    double bend;
    double strip;
    unsigned int ntps;
    vector<int> tps;

    in >> DTClink >> region >> layerdisk >> stubword >> isPSmodule >> isFlipped >> x >> y >> z >> bend >> strip >> ntps;

    for (unsigned int itps = 0; itps < ntps; itps++) {
      int tp;
      in >> tp;
      tps.push_back(tp);
    }

    L1TStub stub(DTClink, region, layerdisk, stubword, isPSmodule, isFlipped, x, y, z, bend, strip, tps);

    in >> tmp;

    double t = std::abs(stub.z()) / stub.r();
    double eta = asinh(t);

    if (std::abs(eta) < 2.6) {
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
  out << "Stubend" << endl;
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
