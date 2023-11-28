#include "L1Trigger/Phase2L1GMT/interface/L1TPhase2GMTBarrelStubProcessor.h"
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

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
  double globalPhi = (sector * 30) + phiS.phi() * 30. / 65535.;
  if (globalPhi < -180)
    globalPhi += 360;
  if (globalPhi > 180)
    globalPhi -= 360;
  globalPhi = globalPhi * M_PI / 180.;
  int phi = int(globalPhi / phiLSB_) + phiOffset_[station - 1];
  int phiB = phiS.phiBend() / phiBFactor_;
  uint tag = phiS.index();
  int bx = phiS.bxNum() - 20;
  int quality = 3;
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
  l1t::MuonStub stub(wheel, sector, station, tfLayer, phi, phiB, tag, bx, quality, eta, 0, 0, 1);
  stub.setOfflineQuantities(globalPhi, float(phiB), eta * etaLSB_, 0.0);
  return stub;
}

l1t::MuonStubCollection L1TPhase2GMTBarrelStubProcessor::makeStubs(const L1Phase2MuDTPhContainer* phiContainer,
                                                                   const L1MuDTChambThContainer* etaContainer) {
  l1t::MuonStubCollection out;
  for (int bx = minBX_; bx <= maxBX_; bx++) {
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
            if (hasEta) {
              out.push_back(buildStub(phiDigi, tseta));
            } else {
              out.push_back(buildStubNoEta(phiDigi));
            }
          }
        }
      }
    }
  }

  if (verbose_) {
    printf("Barrel Stubs\n");
    for (const auto& stub : out)
      printf(
          "Barrel Stub bx=%d TF=%d etaRegion=%d phiRegion=%d depthRegion=%d  coord1=%f,%d coord2=%f,%d eta1=%f,%d "
          "eta2=%f,%d quality=%d etaQuality=%d\n",
          stub.bxNum(),
          stub.tfLayer(),
          stub.etaRegion(),
          stub.phiRegion(),
          stub.depthRegion(),
          stub.offline_coord1(),
          stub.coord1(),
          stub.offline_coord2(),
          stub.coord2(),
          stub.offline_eta1(),
          stub.eta1(),
          stub.offline_eta2(),
          stub.eta2(),
          stub.quality(),
          stub.etaQuality());
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
