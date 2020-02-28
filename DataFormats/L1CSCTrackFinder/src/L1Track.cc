#include "DataFormats/L1CSCTrackFinder/interface/L1Track.h"
#include <iostream>

namespace csc {

  L1Track::L1Track(const csc::L1TrackId& id) : m_name("csc::L1Track") {
    m_endcap = id.endcap();
    m_sector = id.sector();
    m_lphi = 0;
    m_ptAddress = 0;
    m_empty = true;
    setType(2);
    setPtPacked(0);
    m_rank = 0;
    me1_id = 0;
    me2_id = 0;
    me3_id = 0;
    me4_id = 0;
    mb1_id = 0;
    m_me1Tbin = 0;
    m_me2Tbin = 0;
    m_me3Tbin = 0;
    m_me4Tbin = 0;
    m_mbTbin = 0;
    m_output_link = 0;
    m_winner = false;
    m_fr = 0;
    m_se = true;
    m_bx0 = true;
    m_bc0 = true;
  }

  L1Track::L1Track(const csc::L1Track& rhs)
      : L1MuRegionalCand(rhs.type_idx(),
                         rhs.phi_packed(),
                         rhs.eta_packed(),
                         rhs.pt_packed(),
                         rhs.charge_packed(),
                         rhs.charge_valid_packed(),
                         rhs.finehalo_packed(),
                         rhs.quality_packed(),
                         rhs.bx()),
        m_name(rhs.m_name) {
    this->setBx(rhs.bx());
    this->setDataWord(rhs.getDataWord());
    m_name = rhs.m_name;
    this->setType(rhs.type_idx());
    this->setPhiPacked(rhs.phi_packed());
    this->setEtaPacked(rhs.eta_packed());
    this->setPtPacked(rhs.pt_packed());
    this->setChargePacked(rhs.charge_packed());
    this->setChargeValidPacked(rhs.charge_valid_packed());
    this->setFineHaloPacked(rhs.finehalo_packed());
    this->setQualityPacked(rhs.quality_packed());
    m_empty = rhs.m_empty;
    m_lphi = rhs.m_lphi;
    m_endcap = rhs.m_endcap;
    m_sector = rhs.m_sector;
    m_ptAddress = rhs.m_ptAddress;
    m_rank = rhs.m_rank;
    me1_id = rhs.me1_id;
    me2_id = rhs.me2_id;
    me3_id = rhs.me3_id;
    me4_id = rhs.me4_id;
    mb1_id = rhs.mb1_id;
    m_me1Tbin = rhs.m_me1Tbin;
    m_me2Tbin = rhs.m_me2Tbin;
    m_me3Tbin = rhs.m_me3Tbin;
    m_me4Tbin = rhs.m_me4Tbin;
    m_mbTbin = rhs.m_mbTbin;
    m_output_link = rhs.m_output_link;
    m_winner = rhs.m_winner;
    m_fr = rhs.m_fr;
    m_se = rhs.m_se;
    m_bx0 = rhs.m_bx0;
    m_bc0 = rhs.m_bc0;
  }

  L1Track::~L1Track() {}

  const csc::L1Track& L1Track::operator=(const csc::L1Track& rhs) {
    if (this != &rhs) {
      m_fr = rhs.m_fr;
      m_se = rhs.m_se;
      m_bx0 = rhs.m_bx0;
      m_bc0 = rhs.m_bc0;
      m_empty = rhs.m_empty;
      this->setBx(rhs.bx());
      this->setDataWord(rhs.getDataWord());
      m_name = rhs.m_name;
      m_lphi = rhs.m_lphi;
      this->setType(rhs.type_idx());
      this->setPhiPacked(rhs.phi_packed());
      this->setEtaPacked(rhs.eta_packed());
      this->setPtPacked(rhs.pt_packed());
      this->setChargePacked(rhs.charge_packed());
      this->setChargeValidPacked(rhs.charge_valid_packed());
      this->setFineHaloPacked(rhs.finehalo_packed());
      this->setQualityPacked(rhs.quality_packed());
      m_endcap = rhs.m_endcap;
      m_sector = rhs.m_sector;
      m_ptAddress = rhs.m_ptAddress;
      m_rank = rhs.m_rank;
      me1_id = rhs.me1_id;
      me2_id = rhs.me2_id;
      me3_id = rhs.me3_id;
      me4_id = rhs.me4_id;
      mb1_id = rhs.mb1_id;
      m_me1Tbin = rhs.m_me1Tbin;
      m_me2Tbin = rhs.m_me2Tbin;
      m_me3Tbin = rhs.m_me3Tbin;
      m_me4Tbin = rhs.m_me4Tbin;
      m_mbTbin = rhs.m_mbTbin;
      m_output_link = rhs.m_output_link;
      m_winner = rhs.m_winner;
    }
    return *this;
  }

  unsigned L1Track::rank() const { return m_rank; }

  void L1Track::setStationIds(
      const unsigned& me1, const unsigned& me2, const unsigned& me3, const unsigned& me4, const unsigned& mb1) {
    me1_id = me1;
    me2_id = me2;
    me3_id = me3;
    me4_id = me4;
    mb1_id = mb1;
  }

  unsigned L1Track::encodeRank(const unsigned& pt, const unsigned& quality) {
    if (pt == 0)
      return 0;
    return pt | (quality << L1MuRegionalCand::PT_LENGTH);
  }

  void L1Track::decodeRank(const unsigned& rank, unsigned& pt, unsigned& quality) {
    if (rank == 0) {
      quality = 0;
      pt = 0;
    } else {
      quality = rank >> L1MuRegionalCand::PT_LENGTH;
      pt = rank & ((1 << L1MuRegionalCand::PT_LENGTH) - 1);
    }
  }

  bool L1Track::operator>(const csc::L1Track& rhs) const { return (rank() > rhs.rank()); }

  bool L1Track::operator<(const csc::L1Track& rhs) const { return (rank() < rhs.rank()); }

  bool L1Track::operator>=(const csc::L1Track& rhs) const { return (rank() >= rhs.rank()); }

  bool L1Track::operator<=(const csc::L1Track& rhs) const { return (rank() <= rhs.rank()); }

  bool L1Track::operator==(const csc::L1Track& rhs) const { return (rank() == rhs.rank()); }

  bool L1Track::operator!=(const csc::L1Track& rhs) const { return (rank() != rhs.rank()); }

  void L1Track::Print() const {
    if (!empty()) {
      std::cout << "\t  Pt(int): "
                << " " << pt_packed() << " Phi(int): "
                << " " << phi_packed() << " Eta(int): "
                << " " << eta_packed() << " Quality: "
                << " " << quality_packed() << " charge: "
                << " " << charge_packed() << " side: "
                << " " << endcap() << " bx: "
                << " " << BX() << " Winner: "
                << " " << winner() << std::endl;
    } else {
      std::cout << "\t  Empty track!\n";
      std::cout << "\t  Pt(int): "
                << " "
                << "unassigned or zero"
                << " Phi(int): "
                << " " << phi_packed() << " Eta(int): "
                << " " << eta_packed() << " Quality: "
                << " "
                << "unassigned or zero"
                << " charge: "
                << " " << charge_packed() << " side: "
                << " " << endcap() << " bx: "
                << " " << BX() << " Winner: "
                << " " << winner() << std::endl;
    }
  }

  unsigned L1Track::modeExtended(void) const {
    unsigned modeExt = mode();
    unsigned fr = (m_ptAddress >> 21) & 0x1;

    if (modeExt == 11 && fr == 1)
      modeExt = 17;
    else if (modeExt == 12 && fr == 1)
      modeExt = 18;
    else if (modeExt == 14 && fr == 1)
      modeExt = 16;
    return modeExt;
  }
}  // namespace csc
