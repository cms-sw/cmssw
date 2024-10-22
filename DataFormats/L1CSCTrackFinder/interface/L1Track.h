/**
 * \class L1Track (for CSC TrackFinder)
 * \author L. Gray (partial port from ORCA)
 *
 * A L1 Track class for the CSC Track Finder.
 *
 **/

#ifndef L1CSCTrackFinder_L1Track_h
#define L1CSCTrackFinder_L1Track_h

#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1TrackId.h>

class CSCTFSectorProcessor;
class CSCTFUnpacker;
class CSCTFSPCoreLogic;

namespace csc {

  class L1Track : public L1MuRegionalCand {
  public:
    L1Track() : L1MuRegionalCand(), m_name("csc::L1Track") {
      setType(2);
      setPtPacked(0);
    }
    L1Track(const csc::L1TrackId&);
    L1Track(const csc::L1Track&);

    const csc::L1Track& operator=(const csc::L1Track&);

    ~L1Track() override;

    unsigned rank() const;
    void setRank(const unsigned& rank) { m_rank = rank; }

    unsigned localPhi() const { return m_lphi; }
    void setLocalPhi(const unsigned& lphi) { m_lphi = lphi; }

    unsigned me1ID() const { return me1_id; }
    unsigned me2ID() const { return me2_id; }
    unsigned me3ID() const { return me3_id; }
    unsigned me4ID() const { return me4_id; }
    unsigned mb1ID() const { return mb1_id; }

    unsigned endcap() const { return m_endcap; }
    unsigned sector() const { return m_sector; }
    unsigned station() const { return 0; }
    // these next two are needed by the trigger container class
    unsigned subsector() const { return 0; }
    unsigned cscid() const { return 0; }

    int BX() const { return bx(); }

    static unsigned encodeRank(const unsigned& pt, const unsigned& quality);
    static void decodeRank(const unsigned& rank, unsigned& pt, unsigned& quality);

    unsigned ptLUTAddress() const { return m_ptAddress; }
    unsigned front_rear() const { return m_fr; }
    void setPtLUTAddress(const unsigned& adr) { m_ptAddress = adr; }
    void set_front_rear(unsigned fr) { m_fr = fr; }

    unsigned outputLink() const { return m_output_link; }
    void setOutputLink(unsigned oPL) { m_output_link = oPL; }
    bool winner() const { return m_winner; }

    bool operator>(const csc::L1Track&) const;
    bool operator<(const csc::L1Track&) const;
    bool operator>=(const csc::L1Track&) const;
    bool operator<=(const csc::L1Track&) const;
    bool operator==(const csc::L1Track&) const;
    bool operator!=(const csc::L1Track&) const;

    //friend std::ostream& operator<<(std::ostream&, const csc::L1Track&);
    //friend std::ostream& operator<<(std::ostream&, csc::L1Track&);

    /// Only the Unpacker and SectorProcessor should have access to addTrackStub()
    /// This prevents people from adding too many track stubs.

    friend class CSCTFSectorProcessor;  // for track stubs
    friend class ::CSCTFUnpacker;       // for track id bits and track stubs
    friend class ::CSCTFSPCoreLogic;    // for track id bits

    void Print() const;

    // Accessors for some technical information:
    unsigned deltaPhi12(void) const { return m_ptAddress & 0xFF; }
    unsigned deltaPhi23(void) const { return (m_ptAddress >> 8) & 0xF; }
    unsigned addressEta(void) const { return (m_ptAddress >> 12) & 0xF; }
    unsigned mode(void) const { return (m_ptAddress >> 16) & 0xF; }
    bool sign(void) const { return (m_ptAddress >> 20) & 0x1; }
    bool synch_err(void) const { return m_se; }
    bool bx0(void) const { return m_bx0; }
    bool bc0(void) const { return m_bc0; }
    unsigned me1Tbin(void) const { return m_me1Tbin; }
    unsigned me2Tbin(void) const { return m_me2Tbin; }
    unsigned me3Tbin(void) const { return m_me3Tbin; }
    unsigned me4Tbin(void) const { return m_me4Tbin; }
    unsigned mb1Tbin(void) const { return m_mbTbin; }
    void setBits(bool se, bool bx0, bool bc0) {
      m_se = se;
      m_bx0 = bx0;
      m_bc0 = bc0;
    }
    void setTbins(unsigned me1, unsigned me2, unsigned me3, unsigned me4, unsigned mb) {
      m_me1Tbin = me1;
      m_me2Tbin = me2;
      m_me3Tbin = me3;
      m_me4Tbin = me4;
      m_mbTbin = mb;
    }
    void setStationIds(
        const unsigned& me1, const unsigned& me2, const unsigned& me3, const unsigned& me4, const unsigned& mb1);

    unsigned modeExtended(void) const;

  private:
    std::string m_name;
    //L1TrackId m_id; remove this nested class for now... POOL doesn't like it.
    unsigned m_endcap = 0, m_sector = 0;
    //CSCCorrelatedLCTDigiCollection track_stubs;  same as above
    unsigned m_lphi = 0;
    unsigned m_ptAddress = 0;
    unsigned m_fr = 0;
    unsigned me1_id = 0, me2_id = 0, me3_id = 0, me4_id = 0, mb1_id = 0;
    unsigned m_rank = 0;
    unsigned m_output_link = 0;
    bool m_empty = true;
    bool m_winner = false;

    // Technical information:
    bool m_se = true, m_bx0 = true, m_bc0 = true;
    unsigned m_me1Tbin = 0, m_me2Tbin = 0, m_me3Tbin = 0, m_me4Tbin = 0, m_mbTbin = 0;
  };
}  // namespace csc

#endif
