#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>

namespace csc {
  
  L1Track::L1Track(const csc::L1TrackId& id): m_name("csc::L1Track"), m_id(id)
  {
    scale = new L1MuTriggerScales();
    m_sector = 0;
    m_lphi = 0;
    m_ptAddress = 0;
    m_empty = true;
    setType(2); 
    setPtPacked(0);
  }
  
  L1Track::L1Track(const csc::L1Track& rhs) : L1MuRegionalCand(rhs.type_idx(),rhs.phi_packed(),rhs.eta_packed(),
						      rhs.pt_packed(),rhs.charge_packed(),
						      rhs.charge_valid_packed(),rhs.finehalo_packed(),
						      rhs.quality_packed(),rhs.bx()),
				     m_name(rhs.m_name)
  {
    scale = new L1MuTriggerScales();
    m_sector = rhs.m_sector;
    m_empty = rhs.m_empty;
    m_lphi = rhs.m_lphi;
    m_id = rhs.m_id;
    m_ptAddress = rhs.m_ptAddress;
  }

  L1Track::~L1Track()
  {
    if(scale) delete scale;
    scale = NULL;
  }

  const csc::L1Track& L1Track::operator=(const csc::L1Track& rhs)
  {
    if(this != &rhs)
      {
	scale = new L1MuTriggerScales();
	m_sector = rhs.m_sector;
	m_empty = rhs.m_empty;
	this->setBx(rhs.bx());
	this->setDataWord(rhs.getDataWord());
	m_name    = rhs.m_name;
	m_lphi    = rhs.m_lphi;
	this->setType(rhs.type_idx());
	this->setPhiPacked(rhs.phi_packed());
	this->setEtaPacked(rhs.eta_packed());
	this->setPtPacked(rhs.pt_packed());
	this->setChargePacked(rhs.charge_packed());
	this->setChargeValidPacked(rhs.charge_valid_packed());
	this->setFineHaloPacked(rhs.finehalo_packed());
	this->setQualityPacked(rhs.quality_packed());
	m_id      = rhs.m_id;
	m_ptAddress = rhs.m_ptAddress;
      }
    return *this;
  }

  unsigned L1Track::rank() const
  {
    return pt_packed() | quality_packed() << 5;
  }

  float L1Track::ptValueMid() const
  {
    if(pt_packed() >= 31) return scale->getPtScale()->getLowEdge(pt_packed());
    return scale->getPtScale()->getCenter(pt_packed());
  }

  float L1Track:: etaValueLow() const
  {
    return scale->getRegionalEtaScale(type_idx())->getLowEdge(eta_packed());
  }

  float L1Track::phiValueMid() const
  {
    return scale->getPhiScale()->getCenter(phi_packed());
  }

  float L1Track::localPhiValue() const
  {
    return scale->getPhiScale()->getLowEdge(m_lphi);
  }

  void L1Track::addTrackStub(const CSCDetId& id, const CSCCorrelatedLCTDigi& digi)
  {
    m_id.addSegment(id, digi);
  }

  unsigned L1Track::encodeRank(const unsigned& pt, const unsigned& quality)
  {
    if(pt == 0) return 0;
    return pt | (quality << L1MuRegionalCand::PT_LENGTH); 
  }

  void L1Track::decodeRank(const unsigned& rank, unsigned& pt, unsigned& quality)
  {
    if(rank == 0)
      {
	quality = 0;
	pt = 0;
      }
    else
      {
	quality = rank >> L1MuRegionalCand::PT_LENGTH;
	pt = rank & ( (1<<L1MuRegionalCand::PT_LENGTH) - 1);
      }
  }

  unsigned L1Track::encodePt(const double& pt)
  {
    L1MuTriggerScales* tempScales = new L1MuTriggerScales();
    double pT = fabs(pt);
    unsigned i = 0;
    for(i = 0; i < (1<<L1MuRegionalCand::PT_LENGTH) - 1; i++)
      {
	if( pT < tempScales->getPtScale()->getLowEdge(i+1) ||
	    tempScales->getPtScale()->getLowEdge(i+1) == -1 )
	  break;
      }
    delete tempScales;    
    return i;
  } 

  bool L1Track::operator>(const csc::L1Track &rhs) const
  {
    return (m_id.rank() > rhs.m_id.rank()); 
  }

  bool L1Track::operator<(const csc::L1Track& rhs) const
  {
    return (rank() < rhs.rank());
  }

  bool L1Track::operator>=(const csc::L1Track& rhs) const
  {
    return (rank() >= rhs.rank());
  }

  bool L1Track::operator<=(const csc::L1Track& rhs) const
  {
    return (rank() <= rhs.rank());
  }

  bool L1Track::operator==(const csc::L1Track& rhs) const
  {
    return (rank() == rhs.rank());
  }
  
  bool L1Track::operator!=(const csc::L1Track& rhs) const
  {
    return (rank() != rhs.rank());
  }  
}

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
