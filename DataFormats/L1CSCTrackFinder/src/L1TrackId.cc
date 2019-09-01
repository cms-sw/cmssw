#include <DataFormats/L1CSCTrackFinder/interface/L1TrackId.h>

namespace csc {
  L1TrackId::L1TrackId() {}

  L1TrackId::L1TrackId(const unsigned& side, const unsigned& sector) {
    /// Use a fake cscid and station... We just need to know endcap and sector
    id_ = CSCDetId(side, 2, 1, CSCTriggerNumbering::chamberFromTriggerLabels(sector, 0, 2, 1), 0);
  }

  L1TrackId::L1TrackId(const csc::L1TrackId& id) {
    m_rank = id.m_rank;
    m_mode = id.m_mode;
    m_erase = id.m_erase;
    m_overlap = id.m_overlap;
    id_ = id.id_;
    //stubs = id.stubs;  // add stubs back later
  }

  const csc::L1TrackId& L1TrackId::operator=(const csc::L1TrackId& rhs) {
    if (&rhs != this) {
      m_rank = rhs.m_rank;
      m_mode = rhs.m_mode;
      m_erase = rhs.m_erase;
      m_overlap = rhs.m_overlap;
      id_ = rhs.id_;
      //stubs = rhs.stubs;
    }
    return *this;
  }

  bool L1TrackId::sharesHit(const csc::L1TrackId& a_id) const {
    return false;  // finish later
  }

  void L1TrackId::setOverlap(const unsigned& rank) {
    if ((rank == 7) || (rank == 8) || (rank == 9) || (rank == 12) || (rank == 15) || (rank == 19) || (rank == 20) ||
        (rank == 24) || (rank == 25) || (rank == 29) || (rank == 30) || (rank == 32) || (rank == 34) || (rank == 36)) {
      m_overlap = true;
    } else
      m_overlap = false;
  }

  void L1TrackId::setRank(const unsigned& rank) {
    if (rank < (1 << kRankBitWidth))  // rank >= 0, since rank is unsigned
    {
      m_rank = rank;
      setOverlap(rank);
    } else {
      m_rank = 0;
    }
  }

  // Helper function to determine which 2 segments from overlap region
  // track participate in 2-stn Pt assignment, and what mode to use,
  // based on track rank
  // four modes in this order: B1-E1, B1-E2, B2-E1, B2-E2
  // Let's include only mode 2, 4 tracks (7-27-00)
  void L1TrackId::overlapMode(const unsigned& rank, int& mode, int& stnA, int& stnB) {
    switch (rank) {
      case 7:
        stnA = 2;
        stnB = 1;
        mode = 4;
        break;
      case 8:
        stnA = 3;
        stnB = 2;
        mode = 4;
        break;
      case 9:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 12:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 15:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 19:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      case 20:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 24:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      case 25:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 29:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      case 30:
        stnA = 2;
        stnB = 1;
        mode = 2;
        break;
      case 32:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      case 34:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      case 36:
        stnA = 3;
        stnB = 2;
        mode = 2;
        break;
      default:
        // standard case for CSC tracks
        stnA = 1;
        stnB = 2;
        mode = 0;
    }
  }

  unsigned L1TrackId::encodeLUTMode(const unsigned& rank) const {
    int mode;
    switch (rank) {
      case 0:
        mode = 0;
        break;
      case 1:
        mode = 10;
        break;
      case 2:
        mode = 9;
        break;
      case 3:
        mode = 8;
        break;
      case 4:
        mode = 5;
        break;
      case 5:
        mode = 7;
        break;
      case 6:
        mode = 6;
        break;
      case 7:
        mode = 15;
        break;
      case 8:
        mode = 13;
        break;
      case 9:
        mode = 14;
        break;
      case 10:
        mode = 7;
        break;
      case 11:
        mode = 6;
        break;
      case 12:
        mode = 14;
        break;
      case 13:
        mode = 7;
        break;
      case 14:
        mode = 6;
        break;
      case 15:
        mode = 14;
        break;
      case 16:
        mode = 4;
        break;
      case 17:
        mode = 3;
        break;
      case 18:
        mode = 2;
        break;
      case 19:
        mode = 12;
        break;
      case 20:
        mode = 11;
        break;
      case 21:
        mode = 4;
        break;
      case 22:
        mode = 3;
        break;
      case 23:
        mode = 2;
        break;
      case 24:
        mode = 12;
        break;
      case 25:
        mode = 11;
        break;
      case 26:
        mode = 4;
        break;
      case 27:
        mode = 3;
        break;
      case 28:
        mode = 2;
        break;
      case 29:
        mode = 12;
        break;
      case 30:
        mode = 11;
        break;
      case 31:
        mode = 2;
        break;
      case 32:
        mode = 11;
        break;
      case 33:
        mode = 2;
        break;
      case 34:
        mode = 11;
        break;
      case 35:
        mode = 2;
        break;
      case 36:
        mode = 11;
        break;
      default:
        mode = 0;
    }
    return mode;
  }

  bool L1TrackId::hasME1(const unsigned& rank) const {
    bool me = false;
    switch (rank) {
      case 5:
      case 6:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 16:
      case 17:
      case 18:
      case 21:
      case 22:
      case 23:
      case 26:
      case 27:
      case 28:
      case 31:
      case 33:
      case 35:
        me = true;
        break;
      default:
        me = false;
    }
    return me;
  }

  bool L1TrackId::hasME2(const unsigned& rank) const {
    bool me = false;
    switch (rank) {
      case 2:
      case 3:
      case 4:
      case 6:
      case 7:
      case 8:
      case 9:
      case 11:
      case 12:
      case 14:
      case 15:
      case 17:
      case 18:
      case 19:
      case 20:
      case 22:
      case 23:
      case 24:
      case 25:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 34:
      case 35:
      case 36:
        me = true;
        break;
      default:
        me = false;
    }
    return me;
  }

  bool L1TrackId::hasME3(const unsigned& rank) const {
    bool me = false;
    switch (rank) {
      case 1:
      case 3:
      case 4:
      case 5:
      case 10:
      case 13:
      case 16:
      case 18:
      case 21:
      case 23:
      case 26:
      case 28:
      case 31:
      case 33:
      case 35:
        me = true;
        break;
      default:
        me = false;
    }
    return me;
  }

  bool L1TrackId::hasME4(const unsigned& rank) const {
    bool me = false;
    switch (rank) {
      case 1:
      case 2:
      case 4:
      case 16:
      case 17:
      case 21:
      case 22:
      case 26:
      case 27:
      case 31:
      case 33:
      case 35:
        me = true;
        break;
      default:
        me = false;
    }
    return me;
  }

  bool L1TrackId::hasMB1(const unsigned& rank) const {
    bool mb = false;
    switch (rank) {
      case 9:
      case 12:
      case 15:
      case 19:
      case 20:
      case 24:
      case 25:
      case 29:
      case 30:
      case 32:
      case 34:
      case 36:
        mb = true;
        break;
      default:
        mb = false;
    }
    return mb;
  }

}  // namespace csc
