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
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

namespace csc{

  class L1Track : public L1MuRegionalCand
    {
    public:
      L1Track() : L1MuRegionalCand(), scale(new L1MuTriggerScales()), m_name("csc::L1Track") { setType(2); setPtPacked(0); }
      L1Track( const csc::L1TrackId& );
      L1Track( const csc::L1Track& );

      const csc::L1Track& operator=(const csc::L1Track&);

      virtual ~L1Track();

      void setRank(const unsigned&);
      float ptValueMid() const;
      float etaValueLow() const;
      float phiValueMid() const;

      float localPhiValue() const;
      unsigned localPhi() const { return m_lphi; }
      void setLocalPhi(const unsigned& lphi) { m_lphi = lphi; } 

      void addTrackStub(const CSCDetId&, const CSCCorrelatedLCTDigi&);
      
      float side() const { return m_id.side(); }
      unsigned sector() const { return m_sector; }
      void setSector(const unsigned& sector) { m_sector = sector; }

      static unsigned encodeRank(const unsigned& pt, const unsigned& quality);
      static void decodeRank(const unsigned& rank, unsigned& pt, unsigned& quality);
      static unsigned encodePt(const double& pt); 

      bool operator>(const csc::L1Track&) const;
      bool operator<(const csc::L1Track&) const;
      bool operator>=(const csc::L1Track&) const;
      bool operator<=(const csc::L1Track&) const;
      bool operator==(const csc::L1Track&) const;
      bool operator!=(const csc::L1Track&) const;

      friend std::ostream& operator<<(std::ostream&, const csc::L1Track&);
      friend std::ostream& operator<<(std::ostream&, csc::L1Track&);

    private:
      L1MuTriggerScales* scale;
      
      std::string m_name;
      L1TrackId m_id;
      unsigned m_lphi;
      unsigned m_rank;
      unsigned m_ptAddress;
      unsigned m_sector;
      bool m_empty;
     };
}

#endif
