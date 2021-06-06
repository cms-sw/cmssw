#include "L1Trigger/L1TMuon/interface/MicroGMTCancelOutUnit.h"
#include "L1Trigger/L1TMuon/interface/GMTInternalMuon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1t {
  MicroGMTCancelOutUnit::MicroGMTCancelOutUnit() {}

  MicroGMTCancelOutUnit::~MicroGMTCancelOutUnit() {}

  void MicroGMTCancelOutUnit::initialise(L1TMuonGlobalParamsHelper* microGMTParamsHelper) {
    int fwVersion = microGMTParamsHelper->fwVersion();
    m_boPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->bOPosMatchQualLUT(), cancel_t::omtf_bmtf_pos, fwVersion);
    m_boNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->bONegMatchQualLUT(), cancel_t::omtf_bmtf_neg, fwVersion);
    m_foPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->fOPosMatchQualLUT(), cancel_t::omtf_emtf_pos, fwVersion);
    m_foNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->fONegMatchQualLUT(), cancel_t::omtf_emtf_neg, fwVersion);
    m_ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->ovlPosSingleMatchQualLUT(), cancel_t::omtf_omtf_pos, fwVersion);
    m_ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->ovlNegSingleMatchQualLUT(), cancel_t::omtf_omtf_neg, fwVersion);
    m_fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->fwdPosSingleMatchQualLUT(), cancel_t::emtf_emtf_pos, fwVersion);
    m_fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(
        microGMTParamsHelper->fwdNegSingleMatchQualLUT(), cancel_t::emtf_emtf_neg, fwVersion);

    m_lutDict[tftype::omtf_neg + tftype::bmtf * 10] = m_boNegMatchQualLUT;
    m_lutDict[tftype::omtf_pos + tftype::bmtf * 10] = m_boPosMatchQualLUT;
    m_lutDict[tftype::omtf_pos + tftype::omtf_pos * 10] = m_ovlPosSingleMatchQualLUT;
    m_lutDict[tftype::omtf_neg + tftype::omtf_neg * 10] = m_ovlNegSingleMatchQualLUT;
    m_lutDict[tftype::emtf_pos + tftype::emtf_pos * 10] = m_fwdPosSingleMatchQualLUT;
    m_lutDict[tftype::emtf_neg + tftype::emtf_neg * 10] = m_fwdNegSingleMatchQualLUT;
    m_lutDict[tftype::omtf_pos + tftype::emtf_pos * 10] = m_foPosMatchQualLUT;
    m_lutDict[tftype::omtf_neg + tftype::emtf_neg * 10] = m_foNegMatchQualLUT;
  }

  void MicroGMTCancelOutUnit::setCancelOutBits(GMTInternalWedges& wedges, tftype trackFinder, cancelmode mode) {
    std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
    coll1.reserve(3);
    std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
    coll2.reserve(3);
    int maxWedges = 6;
    if (trackFinder == bmtf) {
      maxWedges = 12;
    }
    for (int currentWedge = 0; currentWedge < maxWedges; ++currentWedge) {
      for (const auto& mu : wedges.at(currentWedge)) {
        coll1.push_back(mu);
      }
      // handle wrap around: max "wedge" has to be compared to first "wedge"
      int neighbourWedge = (currentWedge + 1) % maxWedges;
      for (const auto& mu : wedges.at(neighbourWedge)) {
        coll2.push_back(mu);
      }
      if (mode == cancelmode::coordinate) {
        getCoordinateCancelBits(coll2, coll1);  // in case of a tie coll1 muon wins
      } else {
        getTrackAddrCancelBits(mode, coll1, coll2);
      }

      coll1.clear();
      coll2.clear();
    }
  }

  void MicroGMTCancelOutUnit::setCancelOutBitsOverlapBarrel(GMTInternalWedges& omtfSectors,
                                                            GMTInternalWedges& bmtfWedges,
                                                            cancelmode mode) {
    // overlap sector collection
    std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
    coll1.reserve(3);
    // barrel wedge collection with 4 wedges
    std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
    coll2.reserve(12);

    for (int currentSector = 0; currentSector < 6; ++currentSector) {
      for (const auto& omtfMuon : omtfSectors.at(currentSector)) {
        coll1.push_back(omtfMuon);
      }
      // BMTF | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 0  |
      // OMTF |    0    |    1    |    2    |    3    |    4    |    5    |
      // cancel OMTF sector x with corresponding BMTF wedge + the two on either side;
      // e.g. OMTF 0 with BMTF 0, 1, 2, 3, OMTF 2 with BMTF 4, 5, 6, 7 etc.
      for (int i = 0; i < 4; ++i) {
        int currentWedge = (currentSector * 2 + i) % 12;
        for (const auto& bmtfMuon : bmtfWedges.at(currentWedge)) {
          coll2.push_back(bmtfMuon);
        }
      }
      if (mode == cancelmode::coordinate) {
        getCoordinateCancelBits(coll1, coll2);
      } else {
        getTrackAddrCancelBits(mode, coll1, coll2);
      }
      coll1.clear();
      coll2.clear();
    }
  }

  void MicroGMTCancelOutUnit::setCancelOutBitsOverlapEndcap(GMTInternalWedges& omtfSectors,
                                                            GMTInternalWedges& emtfSectors,
                                                            cancelmode mode) {
    // overlap sector collection
    std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
    coll1.reserve(3);
    // endcap sector collection with 3 sectors
    std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
    coll2.reserve(9);

    for (int curOmtfSector = 0; curOmtfSector < 6; ++curOmtfSector) {
      for (const auto& omtfMuon : omtfSectors.at(curOmtfSector)) {
        coll1.push_back(omtfMuon);
      }
      // OMTF |    0    |    1    |    2    |    3    |    4    |    5    |
      // EMTF |    0    |    1    |    2    |    3    |    4    |    5    |
      // cancel OMTF sector x with corresponding EMTF sector + the ones on either side;
      // e.g. OMTF 1 with EMTF 0, 1, 2; OMTF 0 with EMTF 5, 0, 1 etc.
      for (int i = 0; i < 3; ++i) {
        // handling the wrap around: adding 5 because 0 has to be compared to 5
        int curEmtfSector = ((curOmtfSector + 5) + i) % 6;
        for (const auto& emtfMuon : emtfSectors.at(curEmtfSector)) {
          coll2.push_back(emtfMuon);
        }
      }
      if (mode == cancelmode::coordinate) {
        getCoordinateCancelBits(coll1, coll2);
      } else {
        getTrackAddrCancelBits(mode, coll1, coll2);
      }
      coll1.clear();
      coll2.clear();
    }
  }

  void MicroGMTCancelOutUnit::getCoordinateCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1,
                                                      std::vector<std::shared_ptr<GMTInternalMuon>>& coll2) {
    if (coll1.empty() || coll2.empty()) {
      return;
    }
    tftype coll1TfType = (*coll1.begin())->trackFinderType();
    tftype coll2TfType = (*coll2.begin())->trackFinderType();
    if (coll2TfType != tftype::bmtf && coll1TfType % 2 != coll2TfType % 2) {
      edm::LogError("Detector side mismatch")
          << "Overlap-Endcap cancel out between positive and negative detector side attempted. Check eta assignment. "
             "OMTF candidate: TF type: "
          << coll1TfType << ", hwEta: " << (*coll1.begin())->hwEta() << ". EMTF candidate: TF type: " << coll2TfType
          << ", hwEta: " << (*coll2.begin())->hwEta() << ". TF type even: pos. side; odd: neg. side." << std::endl;
      return;
    }

    MicroGMTMatchQualLUT* matchLUT = m_lutDict.at(coll1TfType + coll2TfType * 10).get();

    for (auto mu_w1 = coll1.begin(); mu_w1 != coll1.end(); ++mu_w1) {
      int etaFine1 = (*mu_w1)->hwHF();
      // for EMTF muons set eta fine bit to true since hwHF is the halo bit
      if (coll1TfType == tftype::emtf_pos || coll1TfType == tftype::emtf_neg) {
        etaFine1 = 1;
      }
      for (auto mu_w2 = coll2.begin(); mu_w2 != coll2.end(); ++mu_w2) {
        int etaFine2 = (*mu_w2)->hwHF();
        // for EMTF muons set eta fine bit to true since hwHF is the halo bit
        if (coll2TfType == tftype::emtf_pos || coll2TfType == tftype::emtf_neg) {
          etaFine2 = 1;
        }
        // both muons must have the eta fine bit set in order to use the eta fine part of the LUT
        int etaFine = (int)(etaFine1 > 0 && etaFine2 > 0);

        // The LUT for cancellation takes reduced width phi and eta, we need the LSBs
        int dPhiMask = (1 << matchLUT->getDeltaPhiWidth()) - 1;
        int dEtaMask = (1 << matchLUT->getDeltaEtaWidth()) - 1;

        int dPhi = (*mu_w1)->hwGlobalPhi() - (*mu_w2)->hwGlobalPhi();
        dPhi = std::abs(dPhi);
        if (dPhi > 338)
          dPhi -= 576;  // shifts dPhi to [-pi, pi) in integer scale
        dPhi = std::abs(dPhi);
        int dEta = std::abs((*mu_w1)->hwEta() - (*mu_w2)->hwEta());
        // check first if the delta is within the LSBs that the LUT takes, otherwise the distance
        // is greater than what we want to cancel -> e.g. 31(int) is max => 31*0.01 = 0.31 (rad)
        // LUT takes 5 LSB for dEta and 3 LSB for dPhi
        if (dEta <= dEtaMask && dPhi <= dPhiMask) {
          int match = matchLUT->lookup(etaFine, dEta & dEtaMask, dPhi & dPhiMask);
          if (match == 1) {
            if ((*mu_w1)->hwQual() > (*mu_w2)->hwQual()) {
              (*mu_w2)->setHwCancelBit(1);
            } else {
              (*mu_w1)->setHwCancelBit(1);
            }
          }
        }
      }
    }
  }

  void MicroGMTCancelOutUnit::getTrackAddrCancelBits(cancelmode mode,
                                                     std::vector<std::shared_ptr<GMTInternalMuon>>& coll1,
                                                     std::vector<std::shared_ptr<GMTInternalMuon>>& coll2) {
    if (coll1.empty() || coll2.empty()) {
      return;
    }
    // Address based cancel out for BMTF
    if ((*coll1.begin())->trackFinderType() == tftype::bmtf && (*coll2.begin())->trackFinderType() == tftype::bmtf) {
      if (mode == cancelmode::tracks) {
        getTrackAddrCancelBitsOrigBMTF(coll1, coll2);
      } else if (mode == cancelmode::kftracks) {
        getTrackAddrCancelBitsKfBMTF(coll1, coll2);
      }
      // Address based cancel out for EMTF
    } else if (((*coll1.begin())->trackFinderType() == tftype::emtf_pos &&
                (*coll2.begin())->trackFinderType() == tftype::emtf_pos) ||
               ((*coll1.begin())->trackFinderType() == tftype::emtf_neg &&
                (*coll2.begin())->trackFinderType() == tftype::emtf_neg)) {
      for (auto mu_s1 = coll1.begin(); mu_s1 != coll1.end(); ++mu_s1) {
        std::map<int, int> trkAddr_s1 = (*mu_s1)->origin().trackAddress();
        int me1_ch_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME1Ch];
        int me2_ch_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME2Ch];
        int me3_ch_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME3Ch];
        int me4_ch_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME4Ch];
        if (me1_ch_s1 + me2_ch_s1 + me3_ch_s1 + me4_ch_s1 == 0) {
          continue;
        }
        int me1_seg_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME1Seg];
        int me2_seg_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME2Seg];
        int me3_seg_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME3Seg];
        int me4_seg_s1 = trkAddr_s1[l1t::RegionalMuonCand::emtfAddress::kME4Seg];
        for (auto mu_s2 = coll2.begin(); mu_s2 != coll2.end(); ++mu_s2) {
          std::map<int, int> trkAddr_s2 = (*mu_s2)->origin().trackAddress();
          int me1_ch_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME1Ch];
          int me2_ch_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME2Ch];
          int me3_ch_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME3Ch];
          int me4_ch_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME4Ch];
          if (me1_ch_s2 + me2_ch_s2 + me3_ch_s2 + me4_ch_s2 == 0) {
            continue;
          }
          int me1_seg_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME1Seg];
          int me2_seg_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME2Seg];
          int me3_seg_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME3Seg];
          int me4_seg_s2 = trkAddr_s2[l1t::RegionalMuonCand::emtfAddress::kME4Seg];

          int nMatchedStations = 0;
          if (me1_ch_s2 != 0 && me1_ch_s1 == me1_ch_s2 + 3 && me1_seg_s1 == me1_seg_s2) {
            ++nMatchedStations;
          }
          if (me2_ch_s2 != 0 && me2_ch_s1 == me2_ch_s2 + 2 && me2_seg_s1 == me2_seg_s2) {
            ++nMatchedStations;
          }
          if (me3_ch_s2 != 0 && me3_ch_s1 == me3_ch_s2 + 2 && me3_seg_s1 == me3_seg_s2) {
            ++nMatchedStations;
          }
          if (me4_ch_s2 != 0 && me4_ch_s1 == me4_ch_s2 + 2 && me4_seg_s1 == me4_seg_s2) {
            ++nMatchedStations;
          }

          //std::cout << "Shared hits found: " << nMatchedStations << std::endl;
          if (nMatchedStations > 0) {
            if ((*mu_s1)->origin().hwQual() >= (*mu_s2)->origin().hwQual()) {
              (*mu_s2)->setHwCancelBit(1);
            } else {
              (*mu_s1)->setHwCancelBit(1);
            }
          }
        }
      }
    } else {
      edm::LogError("Cancel out not implemented")
          << "Address based cancel out is currently only implemented for the barrel track finder.";
    }
  }

  void MicroGMTCancelOutUnit::getTrackAddrCancelBitsOrigBMTF(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1,
                                                             std::vector<std::shared_ptr<GMTInternalMuon>>& coll2) {
    for (auto mu_w1 = coll1.begin(); mu_w1 != coll1.end(); ++mu_w1) {
      std::map<int, int> trkAddr_w1 = (*mu_w1)->origin().trackAddress();
      int wheelNum_w1 = trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kWheelNum];
      int wheelSide_w1 = trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kWheelSide];
      std::vector<int> stations_w1;
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat1]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat2]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat3]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat4]);
      //std::cout << "Track address 1: wheelSide (1 == negative side): " << wheelSide_w1 << ", wheelNum: " << wheelNum_w1 << ", stations1234: 0x" << hex << stations_w1[0] << stations_w1[1] << stations_w1[2] << stations_w1[3] << dec << std::endl;

      for (auto mu_w2 = coll2.begin(); mu_w2 != coll2.end(); ++mu_w2) {
        std::map<int, int> trkAddr_w2 = (*mu_w2)->origin().trackAddress();
        int wheelNum_w2 = trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kWheelNum];
        int wheelSide_w2 = trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kWheelSide];
        std::vector<int> stations_w2;
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat1]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat2]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat3]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat4]);
        //std::cout << "Track address 2: wheelSide (1 == negative side): " << wheelSide_w2 << ", wheelNum: " << wheelNum_w2 << ", stations1234: 0x" << hex << stations_w2[0] << stations_w2[1] << stations_w2[2] << stations_w2[3] << dec << std::endl;

        int nMatchedStations = 0;
        // search for duplicates in stations 2-4
        for (int i = 1; i < 4; ++i) {
          if (wheelSide_w1 == wheelSide_w2) {  // both tracks are on the same detector side
            if (wheelNum_w1 == wheelNum_w2) {  // both tracks have the same reference wheel
              if ((stations_w1[i] == 0x0 && stations_w2[i] == 0x2) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0x3) ||
                  (stations_w1[i] == 0x4 && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0x5 && stations_w2[i] == 0x1) ||
                  (stations_w1[i] == 0x8 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x9)) {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 - 1) {  // track 2 is one wheel higher than track 1
              if ((stations_w1[i] == 0x0 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0x4 && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0x5 && stations_w2[i] == 0x9)) {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 + 1) {  // track 2 is one wheel lower than track 1
              if ((stations_w1[i] == 0x8 && stations_w2[i] == 0x2) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0x3) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x1)) {
                ++nMatchedStations;
              }
            }
          } else {
            if (wheelNum_w1 == 0 &&
                wheelNum_w2 == 0) {  // both tracks are on either side of the central wheel (+0 and -0)
              if ((stations_w1[i] == 0x8 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x9)) {
                ++nMatchedStations;
              }
            }
          }
        }
        //std::cout << "Shared hits found: " << nMatchedStations << std::endl;
        if (nMatchedStations > 0) {
          if ((*mu_w1)->origin().hwQual() >= (*mu_w2)->origin().hwQual()) {
            (*mu_w2)->setHwCancelBit(1);
          } else {
            (*mu_w1)->setHwCancelBit(1);
          }
        }
      }
    }
  }

  void MicroGMTCancelOutUnit::getTrackAddrCancelBitsKfBMTF(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1,
                                                           std::vector<std::shared_ptr<GMTInternalMuon>>& coll2) {
    for (auto mu_w1 = coll1.begin(); mu_w1 != coll1.end(); ++mu_w1) {
      std::map<int, int> trkAddr_w1 = (*mu_w1)->origin().trackAddress();
      int wheelNum_w1 = trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kWheelNum];
      int wheelSide_w1 = trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kWheelSide];
      std::vector<int> stations_w1;
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat1]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat2]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat3]);
      stations_w1.push_back(trkAddr_w1[l1t::RegionalMuonCand::bmtfAddress::kStat4]);
      //std::cout << "Track address 1: wheelSide (1 == negative side): " << wheelSide_w1 << ", wheelNum: " << wheelNum_w1 << ", stations1234: 0x" << hex << stations_w1[0] << stations_w1[1] << stations_w1[2] << stations_w1[3] << dec << std::endl;
      //std::cout << "Muon1 eta: " << (*mu_w1)->hwEta() << " phi: " << (*mu_w1)->hwGlobalPhi() << " pT: " << (*mu_w1)->hwPt() << " qual: " << (*mu_w1)->origin().hwQual() << std::endl;

      for (auto mu_w2 = coll2.begin(); mu_w2 != coll2.end(); ++mu_w2) {
        std::map<int, int> trkAddr_w2 = (*mu_w2)->origin().trackAddress();
        int wheelNum_w2 = trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kWheelNum];
        int wheelSide_w2 = trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kWheelSide];
        std::vector<int> stations_w2;
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat1]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat2]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat3]);
        stations_w2.push_back(trkAddr_w2[l1t::RegionalMuonCand::bmtfAddress::kStat4]);
        // std::cout << "Track address 2: wheelSide (1 == negative side): " << wheelSide_w2 << ", wheelNum: " << wheelNum_w2 << ", stations1234: 0x" << hex << stations_w2[0] << stations_w2[1] << stations_w2[2] << stations_w2[3] << dec << std::endl;
        // std::cout << "Muon2 eta: " << (*mu_w2)->hwEta() << " phi: " << (*mu_w2)->hwGlobalPhi() << " pT: " << (*mu_w2)->hwPt() << " qual: " << (*mu_w2)->origin().hwQual() << std::endl;

        int nMatchedStations = 0;
        // search for duplicates in stations 1-3
        for (int i = 0; i < 3; ++i) {
          if (wheelSide_w1 == wheelSide_w2) {  // both tracks are on the same detector side
            if (wheelNum_w1 == wheelNum_w2) {  // both tracks have the same reference wheel
              if ((stations_w1[i] == 0x2 && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0x3 && stations_w2[i] == 0x1) ||
                  (stations_w1[i] == 0x0 && stations_w2[i] == 0x4) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0x5) ||
                  (stations_w1[i] == 0xA && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0xB && stations_w2[i] == 0x9) ||
                  (stations_w1[i] == 0x8 && stations_w2[i] == 0xC) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0xD)) {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 - 1) {  // track 2 is one wheel higher than track 1
              if ((stations_w1[i] == 0xA && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0xB && stations_w2[i] == 0x1) ||
                  (stations_w1[i] == 0x8 && stations_w2[i] == 0x4) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0x5)) {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 + 1) {  // track 2 is one wheel lower than track 1
              if ((stations_w1[i] == 0x2 && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0x3 && stations_w2[i] == 0x9) ||
                  (stations_w1[i] == 0x0 && stations_w2[i] == 0xC) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0xD)) {
                ++nMatchedStations;
              }
            }
          } else {  // If one muon in 0+ and one muon in 0- (0+ and 0- are physically the same wheel), however wheel 0 is not split in kalman algorithm
            if (wheelNum_w1 == 0 && wheelNum_w2 == 1) {
              if ((stations_w1[i] == 0xA && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0xB && stations_w2[i] == 0x1) ||
                  (stations_w1[i] == 0x8 && stations_w2[i] == 0x4) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0x5)) {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == 1 && wheelNum_w2 == 0) {
              if ((stations_w1[i] == 0x2 && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0x3 && stations_w2[i] == 0x9) ||
                  (stations_w1[i] == 0x0 && stations_w2[i] == 0xC) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0xD)) {
                ++nMatchedStations;
              }
            }
          }
        }
        //std::cout << "Shared hits found: " << nMatchedStations << std::endl;
        if (nMatchedStations > 0) {
          if ((*mu_w1)->origin().hwQual() >= (*mu_w2)->origin().hwQual()) {
            (*mu_w2)->setHwCancelBit(1);
          } else {
            (*mu_w1)->setHwCancelBit(1);
          }
        }
      }
    }
  }

}  // namespace l1t
