#include "../interface/MicroGMTCancelOutUnit.h"
#include "L1Trigger/L1TMuon/interface/GMTInternalMuon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1t {
MicroGMTCancelOutUnit::MicroGMTCancelOutUnit ()
{
}

MicroGMTCancelOutUnit::~MicroGMTCancelOutUnit ()
{

}

void
MicroGMTCancelOutUnit::initialise(L1TMuonGlobalParams* microGMTParams) {
    int fwVersion = microGMTParams->fwVersion();
    m_boPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->bOPosMatchQualLUTPath(), microGMTParams->bOPosMatchQualLUTMaxDR(), cancel_t::omtf_bmtf_pos, fwVersion);
    m_boNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->bONegMatchQualLUTPath(), microGMTParams->bONegMatchQualLUTMaxDR(), cancel_t::omtf_bmtf_neg, fwVersion);
    m_foPosMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->fOPosMatchQualLUTPath(), microGMTParams->fOPosMatchQualLUTMaxDR(), cancel_t::omtf_emtf_pos, fwVersion);
    m_foNegMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->fONegMatchQualLUTPath(), microGMTParams->fONegMatchQualLUTMaxDR(), cancel_t::omtf_emtf_neg, fwVersion);
    //m_brlSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->brlSingleMatchQualLUTPath(), microGMTParams->brlSingleMatchQualLUTMaxDR(), cancel_t::bmtf_bmtf, fwVersion);
    m_ovlPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->ovlPosSingleMatchQualLUTPath(), microGMTParams->ovlPosSingleMatchQualLUTMaxDR(), cancel_t::omtf_omtf_pos, fwVersion);
    m_ovlNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->ovlNegSingleMatchQualLUTPath(), microGMTParams->ovlNegSingleMatchQualLUTMaxDR(), cancel_t::omtf_omtf_neg, fwVersion);
    m_fwdPosSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->fwdPosSingleMatchQualLUTPath(), microGMTParams->fwdPosSingleMatchQualLUTMaxDR(), cancel_t::emtf_emtf_pos, fwVersion);
    m_fwdNegSingleMatchQualLUT = l1t::MicroGMTMatchQualLUTFactory::create(microGMTParams->fwdNegSingleMatchQualLUTPath(), microGMTParams->fwdNegSingleMatchQualLUTMaxDR(), cancel_t::emtf_emtf_neg, fwVersion);

    //m_lutDict[tftype::bmtf+tftype::bmtf*10] = m_brlSingleMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::bmtf*10] = m_boNegMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::bmtf*10] = m_boPosMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::omtf_pos*10] = m_ovlPosSingleMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::omtf_neg*10] = m_ovlNegSingleMatchQualLUT;
    m_lutDict[tftype::emtf_pos+tftype::emtf_pos*10] = m_fwdPosSingleMatchQualLUT;
    m_lutDict[tftype::emtf_neg+tftype::emtf_neg*10] = m_fwdNegSingleMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::emtf_pos*10] = m_foPosMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::emtf_neg*10] = m_foNegMatchQualLUT;
}

void
MicroGMTCancelOutUnit::setCancelOutBits(GMTInternalWedges& wedges, tftype trackFinder, cancelmode mode)
{
  std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
  coll1.reserve(3);
  std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
  coll2.reserve(3);
  int maxWedges = 6;
  if (trackFinder == bmtf) {
    maxWedges = 12;
  }
  for (int currentWedge = 0; currentWedge < maxWedges; ++currentWedge) {
    for (auto mu : wedges.at(currentWedge)) {
      coll1.push_back(mu);
    }
    // handle wrap around: max "wedge" has to be compared to first "wedge"
    int neighbourWedge = (currentWedge + 1) % maxWedges;
    for (auto mu : wedges.at(neighbourWedge)) {
      coll2.push_back(mu);
    }
    if (mode == cancelmode::coordinate) {
      getCoordinateCancelBits(coll1, coll2);
    } else {
      getTrackAddrCancelBits(coll1, coll2);
    }

    coll1.clear();
    coll2.clear();
  }
}

void
MicroGMTCancelOutUnit::setCancelOutBitsOverlapBarrel(GMTInternalWedges& omtfSectors, GMTInternalWedges& bmtfWedges, cancelmode mode)
{
  // overlap sector collection
  std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
  coll1.reserve(3);
  // barrel wedge collection with 4 wedges
  std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
  coll2.reserve(12);

  for (int currentSector = 0; currentSector < 6; ++currentSector) {
    for (auto omtfMuon : omtfSectors.at(currentSector)) {
      coll1.push_back(omtfMuon);
    }
    // BMTF | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 0  |
    // OMTF |    0    |    1    |    2    |    3    |    4    |    5    |
    // cancel OMTF sector x with corresponding BMTF wedge + the two on either side;
    // e.g. OMTF 0 with BMTF 0, 1, 2, 3, OMTF 2 with BMTF 4, 5, 6, 7 etc.
    for (int i = 0; i < 4; ++i) {
      int currentWedge = (currentSector * 2 + i) % 12;
      for (auto bmtfMuon : bmtfWedges.at(currentWedge)) {
        coll2.push_back(bmtfMuon);
      }
    }
    if (mode == cancelmode::coordinate) {
      getCoordinateCancelBits(coll1, coll2);
    } else {
      getTrackAddrCancelBits(coll1, coll2);
    }
    coll1.clear();
    coll2.clear();
  }
}

void
MicroGMTCancelOutUnit::setCancelOutBitsOverlapEndcap(GMTInternalWedges& omtfSectors, GMTInternalWedges& emtfSectors, cancelmode mode)
{
  // overlap sector collection
  std::vector<std::shared_ptr<GMTInternalMuon>> coll1;
  coll1.reserve(3);
  // endcap sector collection with 3 sectors
  std::vector<std::shared_ptr<GMTInternalMuon>> coll2;
  coll2.reserve(9);

  for (int curOmtfSector = 0; curOmtfSector < 6; ++curOmtfSector) {
    for (auto omtfMuon : omtfSectors.at(curOmtfSector)) {
      coll1.push_back(omtfMuon);
    }
    // OMTF |    0    |    1    |    2    |    3    |    4    |    5    |
    // EMTF |    0    |    1    |    2    |    3    |    4    |    5    |
    // cancel OMTF sector x with corresponding EMTF sector + the ones on either side;
    // e.g. OMTF 1 with EMTF 0, 1, 2; OMTF 0 with EMTF 5, 0, 1 etc.
    for (int i = 0; i < 3; ++i) {
      // handling the wrap around: adding 5 because 0 has to be compared to 5
      int curEmtfSector = ((curOmtfSector + 5) + i) % 6;
      for (auto emtfMuon : emtfSectors.at(curEmtfSector)) {
        coll2.push_back(emtfMuon);
      }
    }
    if (mode == cancelmode::coordinate) {
      getCoordinateCancelBits(coll1, coll2);
    } else {
      getTrackAddrCancelBits(coll1, coll2);
    }
    coll1.clear();
    coll2.clear();
  }
}

void
MicroGMTCancelOutUnit::getCoordinateCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1, std::vector<std::shared_ptr<GMTInternalMuon>>& coll2)
{
  if (coll1.size() == 0 || coll2.size() == 0) {
    return;
  }
  tftype coll2TfType = (*coll2.begin())->trackFinderType();
  if (coll2TfType != tftype::bmtf && (*coll1.begin())->trackFinderType() % 2 != coll2TfType % 2) {
    edm::LogError("Detector side mismatch") << "Overlap-Endcap cancel out between positive and negative detector side attempted. Check eta assignment. OMTF candidate: TF type: " << (*coll1.begin())->trackFinderType() << ", hwEta: " << (*coll1.begin())->hwEta() << ". EMTF candidate: TF type: " << coll2TfType << ", hwEta: " << (*coll2.begin())->hwEta() << ". TF type even: pos. side; odd: neg. side." << std::endl;
    return;
  }

  MicroGMTMatchQualLUT* matchLUT = m_lutDict.at((*coll1.begin())->trackFinderType()+(*coll2.begin())->trackFinderType()*10).get();

  for (auto mu_w1 = coll1.begin(); mu_w1 != coll1.end(); ++mu_w1) {
    for (auto mu_w2 = coll2.begin(); mu_w2 != coll2.end(); ++mu_w2) {
      // The LUT for cancellation takes reduced width phi and eta, we need the LSBs
      int dPhiMask = (1 << matchLUT->getDeltaPhiWidth()) - 1;
      int dEtaMask = (1 << matchLUT->getDeltaEtaWidth()) - 1;

      // temporary fix to take processor offset into account...
      int dPhi = (*mu_w1)->hwGlobalPhi() - (*mu_w2)->hwGlobalPhi();
      if (dPhi > 338) dPhi -= 576; // shifts dPhi to [-pi, pi) in integer scale
      dPhi = std::abs(dPhi);
      int dEta = std::abs((*mu_w1)->hwEta() - (*mu_w2)->hwEta());
      // check first if the delta is within the LSBs that the LUT takes, otherwise the distance
      // is greater than what we want to cancel -> 15(int) is max => 15*0.01 = 0.15 (rad)
      // LUT takes 4 LSB for dEta and 3 LSB for dPhi
      if (dEta < 16 && dPhi < 8) {
        int match = matchLUT->lookup(dEta & dEtaMask, dPhi & dPhiMask);
        if (match == 1) {
          if((*mu_w1)->hwQual() > (*mu_w2)->hwQual()) {
            (*mu_w2)->setHwCancelBit(1);
          } else {
            (*mu_w1)->setHwCancelBit(1);
          }
        }
      }
    }
  }
}

void
MicroGMTCancelOutUnit::getTrackAddrCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1, std::vector<std::shared_ptr<GMTInternalMuon>>& coll2)
{
  if (coll1.size() == 0 || coll2.size() == 0) {
    return;
  }
  // Address based cancel out is implemented for BMTF only
  if ((*coll1.begin())->trackFinderType() == tftype::bmtf && (*coll2.begin())->trackFinderType() == tftype::bmtf) {
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
          if (wheelSide_w1 == wheelSide_w2) { // both tracks are on the same detector side
            if (wheelNum_w1 == wheelNum_w2) { // both tracks have the same reference wheel
              if ((stations_w1[i] == 0x0 && stations_w2[i] == 0x2) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0x3) ||
                  (stations_w1[i] == 0x4 && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0x5 && stations_w2[i] == 0x1) ||
                  (stations_w1[i] == 0x8 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x9))
              {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 - 1) { // track 2 is one wheel higher than track 1
              if ((stations_w1[i] == 0x0 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x1 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0x4 && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0x5 && stations_w2[i] == 0x9))
              {
                ++nMatchedStations;
              }
            } else if (wheelNum_w1 == wheelNum_w2 + 1) { // track 2 is one wheel lower than track 1
              if ((stations_w1[i] == 0x8 && stations_w2[i] == 0x2) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0x3) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x0) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x1))
              {
                ++nMatchedStations;
              }
            }
          } else {
            if (wheelNum_w1 == 0 && wheelNum_w2 == 0) { // both tracks are on either side of the central wheel (+0 and -0)
              if ((stations_w1[i] == 0x8 && stations_w2[i] == 0xA) ||
                  (stations_w1[i] == 0x9 && stations_w2[i] == 0xB) ||
                  (stations_w1[i] == 0xC && stations_w2[i] == 0x8) ||
                  (stations_w1[i] == 0xD && stations_w2[i] == 0x9))
              {
                ++nMatchedStations;
              }
            }
          }
        }
        //std::cout << "Shared hits found: " << nMatchedStations << std::endl;
        if (nMatchedStations > 0) {
          if ((*mu_w1)->origin().hwQual() > (*mu_w2)->origin().hwQual()) {
            (*mu_w2)->setHwCancelBit(1);
          } else {
            (*mu_w1)->setHwCancelBit(1);
          }
        }
      }
    }
  } else {
    edm::LogError("Cancel out not implemented") << "Address based cancel out is currently only implemented for the barrel track finder.";
  }
}

} // namespace l1t
