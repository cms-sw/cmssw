#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <cmath>
#include <bitset>

using namespace std;
using namespace trklet;

Stub::Stub(Settings const& settings) : settings_(settings) {}

Stub::Stub(L1TStub& stub, Settings const& settings, Globals& globals) : settings_(settings) {
  const string& stubwordhex = stub.stubword();

  const string stubwordbin = convertHexToBin(stubwordhex);

  layerdisk_ = stub.layerdisk();

  int nbendbits = stub.isPSmodule() ? N_BENDBITS_PS : N_BENDBITS_2S;

  int nalphabits = 0;

  int nrbits = settings_.nrbitsstub(layerdisk_);
  int nzbits = settings_.nzbitsstub(layerdisk_);
  int nphibits = settings_.nphibitsstub(layerdisk_);

  if (layerdisk_ >= N_LAYER && !stub.isPSmodule()) {
    nalphabits = settings.nbitsalpha();
    nrbits = 7;
  }

  assert(nbendbits + nalphabits + nrbits + nzbits + nphibits == 36);

  bitset<32> rbits(stubwordbin.substr(0, nrbits));
  bitset<32> zbits(stubwordbin.substr(nrbits, nzbits));
  bitset<32> phibits(stubwordbin.substr(nrbits + nzbits, nphibits));
  bitset<32> alphabits(stubwordbin.substr(nphibits + nzbits + nrbits, nalphabits));
  bitset<32> bendbits(stubwordbin.substr(nphibits + nzbits + nrbits + nalphabits, nbendbits));

  int newbend = bendbits.to_ulong();

  int newr = rbits.to_ulong();
  if (layerdisk_ < N_LAYER) {
    if (newr >= (1 << (nrbits - 1)))
      newr = newr - (1 << nrbits);
  }

  int newz = zbits.to_ulong();
  if (newz >= (1 << (nzbits - 1)))
    newz = newz - (1 << nzbits);

  int newphi = phibits.to_ulong();

  int newalpha = alphabits.to_ulong();
  if (newalpha >= (1 << (nalphabits - 1)))
    newalpha = newalpha - (1 << nalphabits);

  l1tstub_ = &stub;

  bend_.set(newbend, nbendbits, true, __LINE__, __FILE__);

  phi_.set(newphi, nphibits, true, __LINE__, __FILE__);
  phicorr_.set(newphi, nphibits, true, __LINE__, __FILE__);
  bool pos = false;
  if (layerdisk_ >= N_LAYER) {
    pos = true;
    int disk = layerdisk_ - N_LAYER + 1;
    if (stub.z() < 0.0)
      disk = -disk;
    disk_.set(disk, 4, false, __LINE__, __FILE__);
    if (!stub.isPSmodule()) {
      alpha_.set(newalpha, nalphabits, false, __LINE__, __FILE__);
      nrbits = 4;
    }
    int negdisk = (disk < 0) ? 1 : 0;
    negdisk_.set(negdisk, 1, true, __LINE__, __FILE__);
  } else {
    disk_.set(0, 4, false, __LINE__, __FILE__);
    layer_.set(layerdisk_, 3, true, __LINE__, __FILE__);
  }
  r_.set(newr, nrbits, pos, __LINE__, __FILE__);
  z_.set(newz, nzbits, false, __LINE__, __FILE__);

  if (settings.writeMonitorData("StubBend")) {
    unsigned int nsimtrks = globals.event()->nsimtracks();

    for (unsigned int isimtrk = 0; isimtrk < nsimtrks; isimtrk++) {
      const L1SimTrack& simtrk = globals.event()->simtrack(isimtrk);
      if (stub.tpmatch2(simtrk.trackid())) {
        double dr = 0.18;
        double rinv = simtrk.charge() * 0.01 * settings_.c() * settings_.bfield() / simtrk.pt();
        double pitch = settings_.stripPitch(stub.isPSmodule());
        double bend = stub.r() * dr * 0.5 * rinv / pitch;

        globals.ofstream("stubbend.dat") << layerdisk_ << " " << stub.isPSmodule() << " "
                                         << simtrk.pt() * simtrk.charge() << " " << bend << " " << newbend << " "
                                         << settings.benddecode(newbend, layerdisk_, stub.isPSmodule()) << " "
                                         << settings.bendcut(newbend, layerdisk_, stub.isPSmodule()) << endl;
      }
    }
  }
}

FPGAWord Stub::iphivmFineBins(int VMbits, int finebits) const {
  unsigned int finephi = (phicorr_.value() >> (phicorr_.nbits() - VMbits - finebits)) & ((1 << finebits) - 1);
  return FPGAWord(finephi, finebits, true, __LINE__, __FILE__);
}

unsigned int Stub::phiregionaddress() const {
  int iphi = (phicorr_.value() >> (phicorr_.nbits() - settings_.nbitsallstubs(layerdisk())));
  return (iphi << 7) + stubindex_.value();
}

std::string Stub::phiregionaddressstr() const {
  int iphi = (phicorr_.value() >> (phicorr_.nbits() - settings_.nbitsallstubs(layerdisk())));
  FPGAWord phiregion(iphi, 3, true, __LINE__, __FILE__);
  return phiregion.str() + stubindex_.str();
}

void Stub::setAllStubIndex(int nstub) {
  if (nstub >= (1 << N_BITSMEMADDRESS)) {
    if (settings_.debugTracklet())
      edm::LogPrint("Tracklet") << "Warning too large stubindex!";
    nstub = (1 << N_BITSMEMADDRESS) - 1;
  }

  stubindex_.set(nstub, N_BITSMEMADDRESS);
}

void Stub::setPhiCorr(int phiCorr) {
  int iphicorr = phi_.value() - phiCorr;

  if (iphicorr < 0)
    iphicorr = 0;
  if (iphicorr >= (1 << phi_.nbits()))
    iphicorr = (1 << phi_.nbits()) - 1;

  phicorr_.set(iphicorr, phi_.nbits(), true, __LINE__, __FILE__);
}

double Stub::rapprox() const {
  if (disk_.value() == 0) {
    int lr = 1 << (8 - settings_.nrbitsstub(layer_.value()));
    return r_.value() * settings_.kr() * lr + settings_.rmean(layer_.value());
  }
  if (!l1tstub_->isPSmodule()) {
    if (abs(disk_.value()) <= 2)
      return settings_.rDSSinner(r_.value());
    else
      return settings_.rDSSouter(r_.value());
  }
  return r_.value() * settings_.kr();
}

double Stub::zapprox() const {
  if (disk_.value() == 0) {
    int lz = 1;
    if (layer_.value() >= 3) {
      lz = 16;
    }
    return z_.value() * settings_.kz() * lz;
  }
  int sign = 1;
  if (disk_.value() < 0)
    sign = -1;
  if (sign < 0) {
    //Should understand why this is needed to get agreement with integer calculations
    return (z_.value() + 1) * settings_.kz() + sign * settings_.zmean(abs(disk_.value()) - 1);
  } else {
    return z_.value() * settings_.kz() + sign * settings_.zmean(abs(disk_.value()) - 1);
  }
}

double Stub::phiapprox(double phimin, double) const {
  int lphi = 1;
  if (layer_.value() >= 3) {
    lphi = 8;
  }
  return reco::reduceRange(phimin + phi_.value() * settings_.kphi() / lphi);
}

unsigned int Stub::layerdisk() const {
  if (layer_.value() == -1)
    return N_LAYER - 1 + abs(disk_.value());
  return layer_.value();
}
