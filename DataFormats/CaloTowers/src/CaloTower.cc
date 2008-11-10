#include "DataFormats/CaloTowers/interface/CaloTower.h"


CaloTower::CaloTower() {
  emE_=0;
  hadE_=0;
  outerE_=0;
  emLvl1_=0;
  hadLvl1_=0;
}

CaloTower::CaloTower(const CaloTowerDetId& id,
		     double emE, double hadE, double outerE,
		     int ecal_tp, int hcal_tp,
		     const PolarLorentzVector p4,
		     GlobalPoint emPos, GlobalPoint hadPos) : 
  LeafCandidate(0, p4, Point(0,0,0)),  
  id_(id),
  emPosition_(emPos), hadPosition_(hadPos), 
  emE_(emE), hadE_(hadE), outerE_(outerE),
  emLvl1_(ecal_tp), hadLvl1_(hcal_tp) {}


CaloTower::CaloTower(const CaloTowerDetId& id,
		     double emE, double hadE, double outerE,
		     int ecal_tp, int hcal_tp,
		     const LorentzVector p4,
		     GlobalPoint emPos, GlobalPoint hadPos) : 
  LeafCandidate(0, p4, Point(0,0,0)),  
  id_(id),
  emPosition_(emPos), hadPosition_(hadPos),
  emE_(emE), hadE_(hadE), outerE_(outerE),
  emLvl1_(ecal_tp), hadLvl1_(hcal_tp) {}


// recalculated momentum-related quantities wrt user provided vertex Z position


math::PtEtaPhiMLorentzVector CaloTower::hadP4(double vtxZ) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  // note: for now we use the same position for HO as for the other detectors

  double hcalTot;
  if (abs(ieta())<16) hcalTot = (energy() - emE_);
  else hcalTot = hadE_;

  if (hcalTot>0) {
    double ctgTheta = (hadPosition_.z() - vtxZ)/hadPosition_.perp();
    double newEta = asinh(ctgTheta);  
    double pf = 1.0/cosh(newEta);

    newP4 = PolarLorentzVector(hcalTot * pf, newEta, hadPosition_.phi(), 0.0);   
  }
  
  return newP4;
}

math::PtEtaPhiMLorentzVector CaloTower::emP4(double vtxZ) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  if (emE_>0) {
    double ctgTheta = (emPosition_.z() - vtxZ)/emPosition_.perp();
    double newEta = asinh(ctgTheta);  
    double pf = 1.0/cosh(newEta);
  
    newP4 = math::PtEtaPhiMLorentzVector(emE_ * pf, newEta, emPosition_.phi(), 0.0);   
  }
  
  return newP4;
}


// recalculated momentum-related quantities wrt user provided 3D vertex 


math::PtEtaPhiMLorentzVector CaloTower::hadP4(Point v) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  GlobalPoint p(v.x(), v.y(), v.z());

  // note: for now we use the same position for HO as for the other detectors

  double hcalTot;
  if (abs(ieta())<16) hcalTot = (energy() - emE_);
  else hcalTot = hadE_;

  if (hcalTot>0) {
    math::XYZVector dir = math::XYZVector(hadPosition_ - p);
    newP4 = math::PtEtaPhiMLorentzVector(hcalTot * sin(dir.theta()), dir.eta(), dir.phi(), 0.0);  
  }

  return newP4;
}

math::PtEtaPhiMLorentzVector CaloTower::emP4(Point v) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  GlobalPoint p(v.x(), v.y(), v.z());

  if (emE_>0) {
    math::XYZVector dir = math::XYZVector(emPosition_ - p);
    newP4 = math::PtEtaPhiMLorentzVector(emE_ * sin(dir.theta()), dir.eta(), dir.phi(), 0.0);   
  }
  
  return newP4;
}


math::PtEtaPhiMLorentzVector CaloTower::p4(double vtxZ) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  if (abs(ieta())<=29) {
    newP4 += emP4(vtxZ);
    newP4 += hadP4(vtxZ);
  }
  else { // em and had energy in HF are defined in a special way
    double ctgTheta = (emPosition_.z() - vtxZ)/emPosition_.perp(); // em and had positions in HF are forced to be the same
    double newEta = asinh(ctgTheta);  
    double pf = 1.0/cosh(newEta);
    newP4 = math::PtEtaPhiMLorentzVector(p4().energy() * pf, newEta, emPosition_.phi(), 0.0);   
  }

  return newP4;
}


math::PtEtaPhiMLorentzVector CaloTower::p4(Point v) const {

  math::PtEtaPhiMLorentzVector newP4(0,0,0,0);

  if (abs(ieta())<=29) {
    newP4 += emP4(v);
    newP4 += hadP4(v);
  }
  else { // em and had energy in HF are defined in a special way
    GlobalPoint p(v.x(), v.y(), v.z());
    math::XYZVector dir = math::XYZVector(emPosition_ - p); // em and had positions in HF are forced to be the same
    newP4 = math::PtEtaPhiMLorentzVector(p4().energy() * sin(dir.theta()), dir.eta(), dir.phi(), 0.0);   
  }

  return newP4;
}




void CaloTower::addConstituents( const std::vector<DetId>& ids ) {
  constituents_.reserve(constituents_.size()+ids.size());
  constituents_.insert(constituents_.end(),ids.begin(),ids.end());
}

int CaloTower::numCrystals() const {
  if (id_.ietaAbs()>29) return 0;
  
  int nC = 0;
  std::vector<DetId>::const_iterator it = constituents_.begin();
  for (; it!=constituents_.end(); ++it) {
    if (it->det()==DetId::Ecal) ++nC;
  }

  return nC;
}



// Set the CaloTower status word from the number of bad/recovered/problematic
// cells in HCAL and ECAL.

void CaloTower::setCaloTowerStatus(uint numBadHcalChan,uint numBadEcalChan, 
				   uint numRecHcalChan,uint numRecEcalChan,
				   uint numProbHcalChan,uint numProbEcalChan) {

  // the check that the number of bad channels does not exceed 3(25) for
  // hcal (ecal) is performed before setting the staus word in the producer.
  // This is the only place where the flag is set.

  twrStatusWord_ = 0x0;

  twrStatusWord_ |= (numBadEcalChan);
  twrStatusWord_ |= (numRecEcalChan << 5);
  twrStatusWord_ |= (numProbEcalChan << 10); 
  twrStatusWord_ |= (numBadHcalChan << 15);
  twrStatusWord_ |= (numRecHcalChan << 17);
  twrStatusWord_ |= (numProbHcalChan << 19);

  return;
}



std::ostream& operator<<(std::ostream& s, const CaloTower& ct) {
  return s << ct.id() << ":"  << ct.et()
	   << " GeV ET (EM=" << ct.emEt() <<
    " HAD=" << ct.hadEt() << " OUTER=" << ct.outerEt() << ") (" << ct.eta() << "," << ct.phi() << ")";    
}
