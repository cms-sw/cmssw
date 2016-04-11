#ifndef AlgoMuon_H
#define AlgoMuon_H

#include <ostream>

class AlgoMuon{

 public:
  
  // AlgoMuon() : pt(-1.), eta(99.), phi(9999.), disc(-999), bx(0), q(-1), charge(99), refLayer(-1), hits(0) {} // the old one version 
  AlgoMuon() : m_disc(-999), m_phi(9999), m_eta(99), m_refLayer(-1), m_hits(0), m_q(-1), m_bx(0), m_pt(-1), m_charge(99) {}
  AlgoMuon(int disc=-999, int phi=9999, int eta=99, int refLayer=-1, 
              int hits=0, int q=-1, int bx=0, int pt=-1, int charge=99):
              m_disc(disc), m_phi(phi), m_eta(eta), m_refLayer(refLayer), 
              m_hits(hits), m_q(q), m_bx(bx), m_pt(pt), m_charge(charge) {}

  int getDisc() const { return m_disc; }
  int getPhi()  const { return m_phi; }
  int getEta()  const { return m_eta; }
  int getRefLayer() const { return m_refLayer; }
  int getHits() const { return m_hits; }
  int getQ()  const { return m_q; }
  int getBx() const { return m_bx; }
  int getPt() const { return m_pt; }
  int getCharge()   const { return m_charge; }
  int getPhiRHit()  const { return m_phiRHit; }

  void setDisc(int disc) { m_disc = disc; }
  void setPhi(int phi)   { m_phi = phi; }
  void setEta(int eta)   { m_eta = eta; }
  void setRefLayer(int refLayer) { m_refLayer = refLayer; }
  void setHits(int hits) { m_hits = hits; }
  void setQ(int q)    { m_q = q; }
  void setBx(int bx)  { m_bx = bx; }
  void setPt(int pt)  { m_pt = pt; }
  void setCharge(int charge)   { m_charge = charge; }
  void setPhiRHit(int phiRHit) { m_phiRHit = phiRHit; }

  bool isValid() const;  

  bool operator< (const AlgoMuon & o) const;

  friend std::ostream & operator<< (std::ostream &out, const AlgoMuon &o);

 private: 

  int m_disc;
  int m_phi;
  int m_eta;
  int m_refLayer;
  int m_hits;
  int m_q;
  int m_bx; 
  int m_pt;
  int m_charge;
  int m_phiRHit;
  // to add 
  // int m_pdf; 

};
#endif
