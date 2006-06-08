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

      unsigned rank() const;
      void setRank(const unsigned& rank) { m_rank = rank; }
      float ptValueMid() const;
      float etaValueLow() const;
      float phiValueMid() const;

      float localPhiValue() const;
      unsigned localPhi() const { return m_lphi; }
      void setLocalPhi(const unsigned& lphi) { m_lphi = lphi; } 

      //void addTrackStub(const CSCDetId&, const CSCCorrelatedLCTDigi&);
      
      unsigned endcap() const { return m_id.side(); }
      unsigned sector() const { return m_id.sector(); }
      unsigned station() const { return 0; }
      // these next two are needed by the trigger container class
      unsigned subsector() const { return 0; }
      unsigned cscid() const { return 0; } 

      int BX() const { return bx(); }

      void setSector(const unsigned& sector) { m_sector = sector; }
      
      static unsigned encodeRank(const unsigned& pt, const unsigned& quality);
      static void decodeRank(const unsigned& rank, unsigned& pt, unsigned& quality);
      static unsigned encodePt(const double& pt); 

      unsigned ptLUTAddress() const { return m_ptAddress; }
      void setPtLUTAddress(const unsigned& adr) { m_ptAddress = adr; }

      bool operator>(const csc::L1Track&) const;
      bool operator<(const csc::L1Track&) const;
      bool operator>=(const csc::L1Track&) const;
      bool operator<=(const csc::L1Track&) const;
      bool operator==(const csc::L1Track&) const;
      bool operator!=(const csc::L1Track&) const;

      //friend std::ostream& operator<<(std::ostream&, const csc::L1Track&);
      //friend std::ostream& operator<<(std::ostream&, csc::L1Track&);

      void Print() const;

    private:
      L1MuTriggerScales* scale;
      
      std::string m_name;
      L1TrackId m_id;
      unsigned m_lphi;
      unsigned m_ptAddress;
      unsigned m_sector;
      int m_bx;
      unsigned m_rank;
      bool m_empty;
     };
}

/*
ostream& operator << (ostream& output, csc::L1Track& rhs) {
  if (!rhs.empty()) {
    output << "\t  Pt(int): "  << " " << rhs.pt_packed()
           << " Phi(int): " << " " << rhs.phi_packed()
           << " Eta(int): " << " " << rhs.eta_packed()
           << " Quality: "  << " " << rhs.quality_packed()
           << " charge: "   << " " << rhs.chargeValue()
           << " side: "   << " " << rhs.endcap()
           << " bx: "       << " " << rhs.bx()
           << endl;
    output << "\t  Pt(float): "  << " " << rhs.ptValue()
           << " Phi(float): " << " " << rhs.phiValueMid()
           << " Eta(float): " << " " << rhs.etaValueLow();
  }
  else {
    output<<"\t  Empty track!\n";
    output << "\t  Pt(int): "  << " " << "unassigned or zero"
           << " Phi(int): " << " " << rhs.phi_packed()
           << " Eta(int): " << " " << rhs.eta_packed()
           << " Quality: "  << " " << "unassigned or zero"
           << " charge: "   << " " << rhs.chargeValue()
           << " side: "   << " " << rhs.endcap()
           << " bx: "       << " " << rhs.bx()
           << endl;
    output << "\t  Phi(float): " << " " << rhs.phiValueMid()
           << " Eta(float): " << " " << rhs.etaValueLow();
    
  }
  return output;
}

std::ostream& operator << (ostream& output,  const csc::L1Track& rhs) {
  if (!rhs.empty()) {
    output << "\t  Pt(int): "  << " " << rhs.pt_packed()
           << " Phi(int): " << " " << rhs.phi_packed()
           << " Eta(int): " << " " << rhs.eta_packed()
           << " Quality: "  << " " << rhs.quality()
           << " charge: "   << " " << rhs.chargeValue()
           << " side: "   << " " << rhs.endcap()
           << " bx: "       << " " << rhs.bx()
           << endl;
    output << "\t  Pt(float): "  << " " << rhs.ptValue()
           << " Phi(float): " << " " << rhs.phiValueMid()
           << " Eta(float): " << " " << rhs.etaValueLow();
  }
  else {
    output<<"\t  Empty track!\n";
    output << "\t  Pt(int): "  << " " << "unassigned or zero"
           << " Phi(int): " << " " << rhs.phi_packed()
           << " Eta(int): " << " " << rhs.eta_packed()
           << " Quality: "  << " " << "unassigned or zero"
           << " charge: "   << " " << rhs.chargeValue()
           << " side: "   << " " << rhs.endcap()
           << " bx: "       << " " << rhs.bx()
           << endl;
    output << "\t  Phi(float): " << " " << rhs.phiValueMid()
           << " Eta(float): " << " " << rhs.etaValueLow();
  }
  return output;
}
*/
#endif
