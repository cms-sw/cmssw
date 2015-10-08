#include "../interface/MicroGMTCancelOutUnit.h"
#include "DataFormats/L1TMuon/interface/GMTInternalMuon.h"

namespace l1t {
MicroGMTCancelOutUnit::MicroGMTCancelOutUnit (const edm::ParameterSet& iConfig) :
    m_boPosMatchQualLUT(iConfig, "BOPos", cancel_t::omtf_bmtf_pos),
    m_boNegMatchQualLUT(iConfig, "BONeg", cancel_t::omtf_bmtf_neg),
    m_foPosMatchQualLUT(iConfig, "FOPos", cancel_t::omtf_emtf_pos),
    m_foNegMatchQualLUT(iConfig, "FONeg", cancel_t::omtf_emtf_neg),
    m_brlSingleMatchQualLUT(iConfig, "BrlSingle", cancel_t::bmtf_bmtf),
    m_ovlPosSingleMatchQualLUT(iConfig, "OvlPosSingle", cancel_t::omtf_omtf_pos),
    m_ovlNegSingleMatchQualLUT(iConfig, "OvlNegSingle", cancel_t::omtf_omtf_neg),
    m_fwdPosSingleMatchQualLUT(iConfig, "FwdPosSingle", cancel_t::emtf_emtf_pos),
    m_fwdNegSingleMatchQualLUT(iConfig, "FwdNegSingle", cancel_t::emtf_emtf_neg)
  {
    m_lutDict[tftype::bmtf+tftype::bmtf*10] = &m_brlSingleMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::bmtf*10] = &m_boNegMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::bmtf*10] = &m_boPosMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::omtf_pos*10] = &m_ovlPosSingleMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::omtf_neg*10] = &m_ovlNegSingleMatchQualLUT;
    m_lutDict[tftype::emtf_pos+tftype::emtf_pos*10] = &m_fwdPosSingleMatchQualLUT;
    m_lutDict[tftype::emtf_neg+tftype::emtf_neg*10] = &m_fwdNegSingleMatchQualLUT;
    m_lutDict[tftype::omtf_pos+tftype::emtf_pos*10] = &m_foPosMatchQualLUT;
    m_lutDict[tftype::omtf_neg+tftype::emtf_neg*10] = &m_foNegMatchQualLUT;
}

MicroGMTCancelOutUnit::~MicroGMTCancelOutUnit ()
{

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
  MicroGMTMatchQualLUT* matchLUT = m_lutDict.at((*coll1.begin())->trackFinderType()+(*coll2.begin())->trackFinderType()*10);

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
      if (dEta < 15 && dPhi < 15) {
        int match = matchLUT->lookup(dEta & dEtaMask, dPhi & dPhiMask);
        if((*mu_w1)->hwPt() < (*mu_w2)->hwPt() && match == 1) {
          (*mu_w2)->setHwCancelBit(1);
        } else if (match == 1) {
          (*mu_w1)->setHwCancelBit(1);
        }
      }
    }
  }
}

void
MicroGMTCancelOutUnit::getTrackAddrCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>& coll1, std::vector<std::shared_ptr<GMTInternalMuon>>& coll2)
{
  // not entirely clear how to do.. just a hook for now
}

} // namespace l1t
