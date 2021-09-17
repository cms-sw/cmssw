
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanStubProcessor.h"
#include <cmath>
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

#include <iostream>
#include <string>
#include <sstream>

L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor() : minPhiQuality_(0), minBX_(-3), maxBX_(3) {}

L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor(const edm::ParameterSet& iConfig)
    : minPhiQuality_(iConfig.getParameter<int>("minPhiQuality")),
      minBX_(iConfig.getParameter<int>("minBX")),
      maxBX_(iConfig.getParameter<int>("maxBX")),
      eta1_(iConfig.getParameter<std::vector<int> >("cotTheta_1")),
      eta2_(iConfig.getParameter<std::vector<int> >("cotTheta_2")),
      eta3_(iConfig.getParameter<std::vector<int> >("cotTheta_3")),
      disableMasks_(iConfig.getParameter<bool>("disableMasks")),
      verbose_(iConfig.getParameter<int>("verbose")) {}

L1TMuonBarrelKalmanStubProcessor::~L1TMuonBarrelKalmanStubProcessor() {}

bool L1TMuonBarrelKalmanStubProcessor::isGoodPhiStub(const L1MuDTChambPhDigi* stub) {
  if (stub->code() < minPhiQuality_)
    return false;
  return true;
}

L1MuKBMTCombinedStub L1TMuonBarrelKalmanStubProcessor::buildStub(const L1MuDTChambPhDigi& phiS,
                                                                 const L1MuDTChambThDigi* etaS) {
  int wheel = phiS.whNum();
  int sector = phiS.scNum();
  int station = phiS.stNum();
  int phi = phiS.phi();
  int phiB = phiS.phiB();
  bool tag = (phiS.Ts2Tag() == 1);
  int bx = phiS.bxNum();
  int quality = phiS.code();

  //Now full eta
  int qeta1 = 0;
  int qeta2 = 0;
  int eta1 = 255;
  int eta2 = 255;

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
  L1MuKBMTCombinedStub stub(wheel, sector, station, phi, phiB, tag, bx, quality, eta1, eta2, qeta1, qeta2);

  return stub;
}

L1MuKBMTCombinedStub L1TMuonBarrelKalmanStubProcessor::buildStubNoEta(const L1MuDTChambPhDigi& phiS) {
  int wheel = phiS.whNum();
  int sector = phiS.scNum();
  int station = phiS.stNum();
  int phi = phiS.phi();
  int phiB = phiS.phiB();
  bool tag = (phiS.Ts2Tag() == 1);
  int bx = phiS.bxNum();
  int quality = phiS.code();

  //Now full eta
  int qeta1 = 0;
  int qeta2 = 0;
  int eta1 = 7;
  int eta2 = 7;
  L1MuKBMTCombinedStub stub(wheel, sector, station, phi, phiB, tag, bx, quality, eta1, eta2, qeta1, qeta2);

  return stub;
}

L1MuKBMTCombinedStubCollection L1TMuonBarrelKalmanStubProcessor::makeStubs(const L1MuDTChambPhContainer* phiContainer,
                                                                           const L1MuDTChambThContainer* etaContainer,
                                                                           const L1TMuonBarrelParams& params) {
  //get the masks from th standard BMTF setup!
  //    const L1TMuonBarrelParamsRcd& bmtfParamsRcd = setup.get<L1TMuonBarrelParamsRcd>();
  //  bmtfParamsRcd.get(bmtfParamsHandle);
  //  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  //  masks_ =  bmtfParams.l1mudttfmasks;

  //get the masks!
  L1MuDTTFMasks msks = params.l1mudttfmasks;

  L1MuKBMTCombinedStubCollection out;
  for (int bx = minBX_; bx <= maxBX_; bx++) {
    for (int wheel = -2; wheel <= 2; wheel++) {
      for (uint sector = 0; sector < 12; sector++) {
        for (uint station = 1; station < 5; station++) {
          //Have to cook up something for the fact that KMTF doesnt use 2 SP at whel=0
          int lwheel1;
          int lwheel2;
          if (wheel < 0) {
            lwheel1 = wheel - 1;
            lwheel2 = wheel - 1;
          } else if (wheel > 0) {
            lwheel1 = wheel + 1;
            lwheel2 = wheel + 1;
          } else {
            lwheel1 = -1;
            lwheel2 = +1;
          }

          bool phiMask = false;
          bool etaMask = false;
          if (station == 1) {
            phiMask = msks.get_inrec_chdis_st1(lwheel1, sector) | msks.get_inrec_chdis_st1(lwheel2, sector);
            etaMask = msks.get_etsoc_chdis_st1(lwheel1, sector) | msks.get_etsoc_chdis_st1(lwheel2, sector);
          }
          if (station == 2) {
            phiMask = msks.get_inrec_chdis_st2(lwheel1, sector) | msks.get_inrec_chdis_st2(lwheel2, sector);
            etaMask = msks.get_etsoc_chdis_st2(lwheel1, sector) | msks.get_etsoc_chdis_st2(lwheel2, sector);
          }
          if (station == 3) {
            phiMask = msks.get_inrec_chdis_st3(lwheel1, sector) | msks.get_inrec_chdis_st3(lwheel2, sector);
            etaMask = msks.get_etsoc_chdis_st3(lwheel1, sector) | msks.get_etsoc_chdis_st3(lwheel2, sector);
          }
          if (station == 4) {
            phiMask = msks.get_inrec_chdis_st4(lwheel1, sector) | msks.get_inrec_chdis_st4(lwheel2, sector);
          }

          if (disableMasks_) {
            phiMask = false;
            etaMask = false;
          }

          bool hasEta = false;
          L1MuDTChambThDigi const* tseta = etaContainer->chThetaSegm(wheel, station, sector, bx);
          if (tseta && (!etaMask)) {
            hasEta = true;
          }

          //	  printf("Wheel=%d LWheel=%d,%d sector=%d station=%d phiMask=%d etaMask=%d\n",wheel,lwheel1,lwheel2,sector,station,phiMask,etaMask);
          //	  if (abs(wheel)==2 && station==1)
          //	    continue;

          L1MuDTChambPhDigi const* high = phiContainer->chPhiSegm1(wheel, station, sector, bx);
          if (high && (!phiMask)) {
            if (high->code() >= minPhiQuality_) {
              const L1MuDTChambPhDigi& stubPhi = *high;
              if (hasEta) {
                out.push_back(buildStub(stubPhi, tseta));
              } else {
                out.push_back(buildStubNoEta(stubPhi));
              }
            }
          }

          L1MuDTChambPhDigi const* low = phiContainer->chPhiSegm2(wheel, station, sector, bx - 1);
          if (low && !(phiMask)) {
            if (low->code() >= minPhiQuality_) {
              const L1MuDTChambPhDigi& stubPhi = *low;
              if (hasEta) {
                out.push_back(buildStub(stubPhi, tseta));
              } else {
                out.push_back(buildStubNoEta(stubPhi));
              }
            }
          }
        }
      }
    }
  }

  return out;
}

int L1TMuonBarrelKalmanStubProcessor::calculateEta(uint i, int wheel, uint sector, uint station) {
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
    eta = -eta1_[eta + 17];
  else if (station == 2)
    eta = -eta2_[eta + 17];
  else
    eta = -eta3_[eta + 17];

  return eta;
}

void L1TMuonBarrelKalmanStubProcessor::makeInputPattern(const L1MuDTChambPhContainer* phiContainer,
                                                        const L1MuDTChambThContainer* etaContainer,
                                                        int sector) {
  int previousSector;
  int nextSector;

  if (sector == 11)
    nextSector = 0;
  else
    nextSector = sector + 1;

  if (sector == 0)
    previousSector = 11;
  else
    previousSector = sector - 1;

  ostringstream os;
  os << "I " << sector << " ";

  bool hasStub = false;

  //start by previous sector
  for (int wheel = -2; wheel < 3; ++wheel) {
    const L1MuDTChambPhDigi* seg1 = phiContainer->chPhiSegm1(wheel, 1, previousSector, 0);
    if (seg1 && seg1->phi() > 111) {
      os << seg1->phi() - 2144 << " " << seg1->phiB() << " " << seg1->code() << " 1 4 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg2 = phiContainer->chPhiSegm1(wheel, 2, previousSector, 0);
    if (seg2 && seg2->phi() > 111) {
      os << seg2->phi() - 2144 << " " << seg2->phiB() << " " << seg2->code() << " 1 4 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg3 = phiContainer->chPhiSegm1(wheel, 3, previousSector, 0);
    if (seg3 && seg3->phi() > 111) {
      os << seg3->phi() - 2144 << " " << seg3->phiB() << " " << seg3->code() << " 1 4 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg4 = phiContainer->chPhiSegm1(wheel, 4, previousSector, 0);
    if (seg4 && seg4->phi() > 111) {
      os << seg4->phi() - 2144 << " " << seg4->phiB() << " " << seg4->code() << " 1 4 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    const L1MuDTChambThDigi* eta1 = etaContainer->chThetaSegm(wheel, 1, previousSector, 0);
    if (eta1) {
      int etaPos = eta1->position(0) + (eta1->position(1) << 1) + (eta1->position(2) << 2) + (eta1->position(3) << 3) +
                   (eta1->position(4) << 4) + (eta1->position(5) << 5) + (eta1->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambThDigi* eta2 = etaContainer->chThetaSegm(wheel, 2, previousSector, 0);
    if (eta2) {
      int etaPos = eta2->position(0) + (eta2->position(1) << 1) + (eta2->position(2) << 2) + (eta2->position(3) << 3) +
                   (eta2->position(4) << 4) + (eta2->position(5) << 5) + (eta2->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }

    const L1MuDTChambThDigi* eta3 = etaContainer->chThetaSegm(wheel, 3, previousSector, 0);
    if (eta3) {
      int etaPos = eta3->position(0) + (eta3->position(1) << 1) + (eta3->position(2) << 2) + (eta3->position(3) << 3) +
                   (eta3->position(4) << 4) + (eta3->position(5) << 5) + (eta3->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambPhDigi* seg5 = phiContainer->chPhiSegm2(wheel, 1, previousSector, -1);
    if (seg5 && seg5->phi() > 111) {
      os << seg5->phi() - 2144 << " " << seg5->phiB() << " " << seg5->code() << " 1 5 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    const L1MuDTChambPhDigi* seg6 = phiContainer->chPhiSegm2(wheel, 2, previousSector, -1);
    if (seg6 && seg6->phi() > 111) {
      os << seg6->phi() - 2144 << " " << seg6->phiB() << " " << seg6->code() << " 1 5 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg7 = phiContainer->chPhiSegm2(wheel, 3, previousSector, -1);
    if (seg7 && seg7->phi() > 111) {
      os << seg7->phi() - 2144 << " " << seg7->phiB() << " " << seg7->code() << " 1 5 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    //const L1MuDTChambPhDigi* seg8 = phiContainer->chPhiSegm2(wheel, 4, previousSector, 1);
    const L1MuDTChambPhDigi* seg8 = phiContainer->chPhiSegm2(wheel, 4, previousSector, -1);
    if (seg8 && seg8->phi() > 111) {
      os << seg8->phi() - 2144 << " " << seg8->phiB() << " " << seg8->code() << " 1 5 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    if (eta1) {
      int etaPos = eta1->quality(0) + (eta1->quality(1) << 1) + (eta1->quality(2) << 2) + (eta1->quality(3) << 3) +
                   (eta1->quality(4) << 4) + (eta1->quality(5) << 5) + (eta1->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    if (eta2) {
      int etaPos = eta2->quality(0) + (eta2->quality(1) << 1) + (eta2->quality(2) << 2) + (eta2->quality(3) << 3) +
                   (eta2->quality(4) << 4) + (eta2->quality(5) << 5) + (eta2->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    if (eta3) {
      int etaPos = eta3->quality(0) + (eta3->quality(1) << 1) + (eta3->quality(2) << 2) + (eta3->quality(3) << 3) +
                   (eta3->quality(4) << 4) + (eta3->quality(5) << 5) + (eta3->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }

    //////////////////////CENTRAL SECTOR
    const L1MuDTChambPhDigi* seg9 = phiContainer->chPhiSegm1(wheel, 1, sector, 0);
    if (seg9) {
      os << seg9->phi() << " " << seg9->phiB() << " " << seg9->code() << " 1 0 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg10 = phiContainer->chPhiSegm1(wheel, 2, sector, 0);
    if (seg10) {
      hasStub = true;
      os << seg10->phi() << " " << seg10->phiB() << " " << seg10->code() << " 1 0 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg11 = phiContainer->chPhiSegm1(wheel, 3, sector, 0);
    if (seg11) {
      hasStub = true;
      os << seg11->phi() << " " << seg11->phiB() << " " << seg11->code() << " 1 0 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg12 = phiContainer->chPhiSegm1(wheel, 4, sector, 0);
    if (seg12) {
      hasStub = true;
      os << seg12->phi() << " " << seg12->phiB() << " " << seg12->code() << " 1 2 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambThDigi* eta4 = etaContainer->chThetaSegm(wheel, 1, sector, 0);
    if (eta4) {
      int etaPos = eta4->position(0) + (eta4->position(1) << 1) + (eta4->position(2) << 2) + (eta4->position(3) << 3) +
                   (eta4->position(4) << 4) + (eta4->position(5) << 5) + (eta4->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }

    const L1MuDTChambThDigi* eta5 = etaContainer->chThetaSegm(wheel, 2, sector, 0);
    if (eta5) {
      int etaPos = eta5->position(0) + (eta5->position(1) << 1) + (eta5->position(2) << 2) + (eta5->position(3) << 3) +
                   (eta5->position(4) << 4) + (eta5->position(5) << 5) + (eta5->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambThDigi* eta6 = etaContainer->chThetaSegm(wheel, 3, sector, 0);
    if (eta6) {
      int etaPos = eta6->position(0) + (eta6->position(1) << 1) + (eta6->position(2) << 2) + (eta6->position(3) << 3) +
                   (eta6->position(4) << 4) + (eta6->position(5) << 5) + (eta6->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }

    const L1MuDTChambPhDigi* seg13 = phiContainer->chPhiSegm2(wheel, 1, sector, -1);
    if (seg13) {
      os << seg13->phi() << " " << seg13->phiB() << " " << seg13->code() << " 1 1 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    const L1MuDTChambPhDigi* seg14 = phiContainer->chPhiSegm2(wheel, 2, sector, -1);
    if (seg14) {
      hasStub = true;

      os << seg14->phi() << " " << seg14->phiB() << " " << seg14->code() << " 1 1 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    const L1MuDTChambPhDigi* seg15 = phiContainer->chPhiSegm2(wheel, 3, sector, -1);
    if (seg15) {
      hasStub = true;

      os << seg15->phi() << " " << seg15->phiB() << " " << seg15->code() << " 1 1 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg16 = phiContainer->chPhiSegm2(wheel, 4, sector, -1);
    if (seg16) {
      hasStub = true;

      os << seg16->phi() << " " << seg16->phiB() << " " << seg16->code() << " 1 1 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    if (eta4) {
      int etaPos = eta4->quality(0) + (eta4->quality(1) << 1) + (eta4->quality(2) << 2) + (eta4->quality(3) << 3) +
                   (eta4->quality(4) << 4) + (eta4->quality(5) << 5) + (eta4->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }

    if (eta5) {
      int etaPos = eta5->quality(0) + (eta5->quality(1) << 1) + (eta5->quality(2) << 2) + (eta5->quality(3) << 3) +
                   (eta5->quality(4) << 4) + (eta5->quality(5) << 5) + (eta5->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    if (eta6) {
      int etaPos = eta6->quality(0) + (eta6->quality(1) << 1) + (eta6->quality(2) << 2) + (eta6->quality(3) << 3) +
                   (eta6->quality(4) << 4) + (eta6->quality(5) << 5) + (eta6->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    /////NEXT SECTOR/////
    const L1MuDTChambPhDigi* seg17 = phiContainer->chPhiSegm1(wheel, 1, nextSector, 0);
    if (seg17 && seg17->phi() < -112) {
      os << seg17->phi() + 2144 << " " << seg17->phiB() << " " << seg17->code() << " 1 2 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg18 = phiContainer->chPhiSegm1(wheel, 2, nextSector, 0);
    if (seg18 && seg18->phi() < -112) {
      os << seg18->phi() + 2144 << " " << seg18->phiB() << " " << seg18->code() << " 1 2 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg19 = phiContainer->chPhiSegm1(wheel, 3, nextSector, 0);
    if (seg19 && seg19->phi() < -112) {
      os << seg19->phi() + 2144 << " " << seg19->phiB() << " " << seg19->code() << " 1 2 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg20 = phiContainer->chPhiSegm1(wheel, 4, nextSector, 0);
    if (seg20 && seg20->phi() < -112) {
      os << seg20->phi() + 2144 << " " << seg20->phiB() << " " << seg20->code() << " 1 2 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambThDigi* eta7 = etaContainer->chThetaSegm(wheel, 1, nextSector, 0);
    if (eta7) {
      int etaPos = eta7->position(0) + (eta7->position(1) << 1) + (eta7->position(2) << 2) + (eta7->position(3) << 3) +
                   (eta7->position(4) << 4) + (eta7->position(5) << 5) + (eta7->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambThDigi* eta8 = etaContainer->chThetaSegm(wheel, 2, nextSector, 0);
    if (eta8) {
      int etaPos = eta8->position(0) + (eta8->position(1) << 1) + (eta8->position(2) << 2) + (eta8->position(3) << 3) +
                   (eta8->position(4) << 4) + (eta8->position(5) << 5) + (eta8->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambThDigi* eta9 = etaContainer->chThetaSegm(wheel, 3, nextSector, 0);
    if (eta9) {
      int etaPos = eta9->position(0) + (eta9->position(1) << 1) + (eta9->position(2) << 2) + (eta9->position(3) << 3) +
                   (eta9->position(4) << 4) + (eta9->position(5) << 5) + (eta9->position(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    const L1MuDTChambPhDigi* seg21 = phiContainer->chPhiSegm2(wheel, 1, nextSector, -1);
    if (seg21 && seg21->phi() < -112) {
      os << seg21->phi() + 2144 << " " << seg21->phiB() << " " << seg21->code() << " 1 3 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }

    const L1MuDTChambPhDigi* seg22 = phiContainer->chPhiSegm2(wheel, 2, nextSector, -1);
    if (seg22 && seg22->phi() < -112) {
      os << seg22->phi() + 2144 << " " << seg22->phiB() << " " << seg22->code() << " 1 3 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg23 = phiContainer->chPhiSegm2(wheel, 3, nextSector, -1);
    if (seg23 && seg23->phi() < -112) {
      os << seg23->phi() + 2144 << " " << seg23->phiB() << " " << seg23->code() << " 1 3 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    const L1MuDTChambPhDigi* seg24 = phiContainer->chPhiSegm2(wheel, 4, nextSector, -1);
    if (seg24 && seg24->phi() < -112) {
      os << seg24->phi() + 2144 << " " << seg24->phiB() << " " << seg24->code() << " 1 3 ";
    } else {
      os << "-2048 0 0 0 15 ";
    }
    if (eta7) {
      int etaPos = eta7->quality(0) + (eta7->quality(1) << 1) + (eta7->quality(2) << 2) + (eta7->quality(3) << 3) +
                   (eta7->quality(4) << 4) + (eta7->quality(5) << 5) + (eta7->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    if (eta8) {
      int etaPos = eta8->quality(0) + (eta8->quality(1) << 1) + (eta8->quality(2) << 2) + (eta8->quality(3) << 3) +
                   (eta8->quality(4) << 4) + (eta8->quality(5) << 5) + (eta8->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
    if (eta9) {
      int etaPos = eta9->quality(0) + (eta9->quality(1) << 1) + (eta9->quality(2) << 2) + (eta9->quality(3) << 3) +
                   (eta9->quality(4) << 4) + (eta9->quality(5) << 5) + (eta9->quality(6) << 6);
      os << etaPos << " ";
    } else {
      os << " 0 ";
    }
  }
  if (hasStub) {
    std::cout << os.str() << std::endl;
  }
}
