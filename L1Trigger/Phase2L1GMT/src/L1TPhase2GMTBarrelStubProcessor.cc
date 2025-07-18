#include "L1Trigger/Phase2L1GMT/interface/L1TPhase2GMTBarrelStubProcessor.h"
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ap_int.h>

L1TPhase2GMTBarrelStubProcessor::L1TPhase2GMTBarrelStubProcessor() : minPhiQuality_(0), minBX_(-3), maxBX_(3) {}

L1TPhase2GMTBarrelStubProcessor::L1TPhase2GMTBarrelStubProcessor(const edm::ParameterSet& iConfig)
    : minPhiQuality_(iConfig.getParameter<int>("minPhiQuality")),
      minBX_(iConfig.getParameter<int>("minBX")),
      maxBX_(iConfig.getParameter<int>("maxBX")),
      eta1_(iConfig.getParameter<std::vector<int> >("eta_1")),
      eta2_(iConfig.getParameter<std::vector<int> >("eta_2")),
      eta3_(iConfig.getParameter<std::vector<int> >("eta_3")),
      coarseEta1_(iConfig.getParameter<std::vector<int> >("coarseEta_1")),
      coarseEta2_(iConfig.getParameter<std::vector<int> >("coarseEta_2")),
      coarseEta3_(iConfig.getParameter<std::vector<int> >("coarseEta_3")),
      coarseEta4_(iConfig.getParameter<std::vector<int> >("coarseEta_4")),
      phiOffset_(iConfig.getParameter<std::vector<int> >("phiOffset")),
      phiBFactor_(iConfig.getParameter<int>("phiBDivider")),
      verbose_(iConfig.getParameter<int>("verbose")),
      phiLSB_(iConfig.getParameter<double>("phiLSB")),
      etaLSB_(iConfig.getParameter<double>("etaLSB")) {}

L1TPhase2GMTBarrelStubProcessor::~L1TPhase2GMTBarrelStubProcessor() {}

l1t::MuonStub L1TPhase2GMTBarrelStubProcessor::buildStub(const L1Phase2MuDTPhDigi& phiS,
                                                         const L1MuDTChambThDigi* etaS) {
  l1t::MuonStub stub = buildStubNoEta(phiS);

  //Now full eta
  int qeta1 = -16384;
  int qeta2 = -16384;
  int eta1 = -16384;
  int eta2 = -16384;

  bool hasEta = false;
  for (uint i = 0; i < 7; ++i) {
    if (etaS->position(i) == 0)
      continue;
    if (!hasEta) {
      hasEta = true;
      eta1 = calculateEta(i, etaS->whNum(), etaS->scNum(), etaS->stNum());

      if (etaS->quality(i) == 1)
        qeta1 = 2;
      else
        qeta1 = 1;
    } else {
      eta2 = calculateEta(i, etaS->whNum(), etaS->scNum(), etaS->stNum());
      if (etaS->quality(i) == 1)
        qeta2 = 2;
      else
        qeta2 = 1;
    }
  }

  if (qeta2 > 0) {  //both stubs->average
    stub.setEta(eta1, eta2, 3);
    stub.setOfflineQuantities(stub.offline_coord1(), stub.offline_coord2(), eta1 * etaLSB_, eta2 * etaLSB_);

  } else if (qeta1 > 0) {  //Good single stub
    stub.setEta(eta1, 0, 1);
    stub.setOfflineQuantities(stub.offline_coord1(), stub.offline_coord2(), eta1 * etaLSB_, 0.0);
  }

  return stub;
}

l1t::MuonStub L1TPhase2GMTBarrelStubProcessor::buildStubNoEta(const L1Phase2MuDTPhDigi& phiS) {
  int wheel = phiS.whNum();
  int abswheel = fabs(phiS.whNum());
  int sign = wheel > 0 ? 1 : -1;
  int sector = phiS.scNum();
  int station = phiS.stNum();

  ap_uint<18> normalization0 = sector * ap_uint<15>(21845);
  ap_int<18> normalization1 = ap_int<18>(ap_int<17>(phiS.phi()) * ap_ufixed<8, 0>(0.3183));
  ap_int<18> kmtf_phi = ap_int<18>(normalization0 + normalization1);
  int phi = int(kmtf_phi);
  float globalPhi = phi * M_PI / (1 << 17);

  //  double globalPhi = (sector * 30) + phiS.phi() * 30. / 65535.;
  int tag = phiS.index();

  int bx = phiS.bxNum() - 20;
  int quality = phiS.quality();
  uint tfLayer = phiS.stNum() - 1;
  int eta = -16384;
  if (station == 1) {
    eta = coarseEta1_[abswheel];
  } else if (station == 2) {
    eta = coarseEta2_[abswheel];
  } else if (station == 3) {
    eta = coarseEta3_[abswheel];
  } else if (station == 4) {
    eta = coarseEta4_[abswheel];
  }

  //override!!!
  //  eta=abswheel;

  //Now full eta

  eta = eta * sign;
  l1t::MuonStub stub(wheel, sector, station, tfLayer, phi, phiS.phiBend(), tag, bx, quality, eta, 0, 0, 1);

  stub.setOfflineQuantities(globalPhi, float(phiS.phiBend() * 0.49e-3), eta * etaLSB_, 0.0);
  return stub;
}

l1t::MuonStubCollection L1TPhase2GMTBarrelStubProcessor::makeStubs(const L1Phase2MuDTPhContainer* phiContainer,
                                                                   const L1MuDTChambThContainer* etaContainer) {
  l1t::MuonStubCollection out;
  for (int bx = minBX_; bx <= maxBX_; bx++) {
    ostringstream os;
    if (verbose_ == 2)
      os << "PATTERN ";
    for (int wheel = -2; wheel <= 2; wheel++) {
      for (int sector = 0; sector < 12; sector++) {
        for (int station = 1; station < 5; station++) {
          bool hasEta = false;
          const L1MuDTChambThDigi* tseta = etaContainer->chThetaSegm(wheel, station, sector, bx);
          if (tseta != nullptr) {
            hasEta = true;
          }

          for (const auto& phiDigi : *phiContainer->getContainer()) {
            if ((phiDigi.bxNum() - 20) != bx || phiDigi.whNum() != wheel || phiDigi.scNum() != sector ||
                phiDigi.stNum() != station)
              continue;
            if (phiDigi.quality() < minPhiQuality_)
              continue;

            if (verbose_ == 2) {
              ap_uint<64> wphi = ap_uint<17>(phiDigi.phi());
              ap_uint<64> wphib = ap_uint<13>(phiDigi.phiBend());
              ap_uint<64> wr1 = ap_uint<21>(0);
              ap_uint<64> wq = ap_uint<4>(phiDigi.quality());
              ap_uint<64> wr2 = ap_uint<9>(0);
              ap_uint<64> sN = 0;
              sN = sN | wphi;
              sN = sN | (wphib << 17);
              sN = sN | (wr1 << 30);
              sN = sN | (wq << 51);
              sN = sN | (wr2 << 55);
              os << std::setw(0) << std::dec << sector << " " << wheel << " " << station << " ";
              os << std::uppercase << std::setfill('0') << std::setw(16) << std::hex << uint64_t(sN) << " ";
            }

            if (hasEta) {
              out.push_back(buildStub(phiDigi, tseta));
            } else {
              out.push_back(buildStubNoEta(phiDigi));
            }
          }
        }
      }
    }
    if (verbose_ == 2)
      edm::LogInfo("BarrelStub") << os.str() << std::endl;
  }

  if (verbose_) {
    edm::LogInfo("BarrelStub") << "Barrel Stubs";
    for (const auto& stub : out)
      edm::LogInfo("BarrelStub") << "Barrel Stub bx=" << stub.bxNum() << " TF=" << stub.tfLayer()
                                 << " etaRegion=" << stub.etaRegion() << " phiRegion=" << stub.phiRegion()
                                 << " depthRegion=" << stub.depthRegion() << "  coord1=" << stub.offline_coord1() << ","
                                 << stub.coord1() << " coord2=" << stub.offline_coord2() << "," << stub.coord2()
                                 << " eta1=" << stub.offline_eta1() << "," << stub.eta1()
                                 << " eta2=" << stub.offline_eta2() << "," << stub.eta2()
                                 << " quality=" << stub.quality() << " etaQuality=" << stub.etaQuality();
  }

  return out;
}

int L1TPhase2GMTBarrelStubProcessor::calculateEta(uint i, int wheel, uint sector, uint station) {
  int eta = 0;
  if (wheel > 0) {
    eta = 7 * wheel + 3 - i;
  } else if (wheel < 0) {
    eta = 7 * wheel + i - 3;
  } else {
    if (sector == 0 || sector == 3 || sector == 4 || sector == 7 || sector == 8 || sector == 11)
      eta = i - 3;
    else
      eta = 3 - i;
  }

  if (station == 1)
    eta = eta1_[eta + 17];
  else if (station == 2)
    eta = eta2_[eta + 17];
  else
    eta = eta3_[eta + 17];

  return eta;
}
