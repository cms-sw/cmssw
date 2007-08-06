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

namespace csc{

  class L1Track : public L1MuRegionalCand
    {
    public:
      L1Track() : L1MuRegionalCand(), m_name("csc::L1Track") { setType(2); setPtPacked(0); }
      L1Track( const csc::L1TrackId& );
      L1Track( const csc::L1Track& );

      const csc::L1Track& operator=(const csc::L1Track&);

      virtual ~L1Track();

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
      void setPtLUTAddress(const unsigned& adr) { m_ptAddress = adr; }

      unsigned outputLink() const {return m_output_link;}
      bool winner() const {return m_winner;}

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

      friend class CSCTFSectorProcessor; // for track stubs
      friend class CSCTFUnpacker; // for track id bits and track stubs
      friend class CSCTFSPCoreLogic; // for track id bits

      void Print() const;

    private:

      std::string m_name;
      //L1TrackId m_id; remove this nested class for now... POOL doesn't like it.
      unsigned m_endcap, m_sector;
      //CSCCorrelatedLCTDigiCollection track_stubs;  same as above
      unsigned m_lphi;
      unsigned m_ptAddress;
      unsigned me1_id, me2_id, me3_id, me4_id, mb1_id;
      unsigned m_rank;
      unsigned m_output_link;
      bool m_empty;
      bool m_winner;

      void setStationIds(const unsigned& me1, const unsigned& me2,
			 const unsigned& me3, const unsigned& me4,
			 const unsigned& mb1);
     };
}

#endif
