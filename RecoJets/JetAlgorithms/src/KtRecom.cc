#include "RecoJets/JetAlgorithms/interface/KtRecom.h"
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#include <string>

namespace KtJet {

KtRecom* getRecomScheme(int recom) {
  if (recom == 1)   return new KtRecomE();
  else if (recom == 2)   return new KtRecomPt();
  else if (recom == 3)   return new KtRecomPt2();
  else if (recom == 4)   return new KtRecomEt();
  else if (recom == 5)   return new KtRecomEt2();
  else{
    std::cout << "[Jets] WARNING, unreconised recombination scheme specified!" << std::endl;
    std::cout << "[Jets] Recombination Scheme set to KtRecomE" << std::endl;
    return new KtRecomE();
  }
}


KtRecomE::KtRecomE() : m_name("E") {}
  //KtRecomE::~KtRecomE() {}
std::string KtRecomE::name() const {return m_name;}

HepLorentzVector KtRecomE::operator()(const HepLorentzVector &a, const HepLorentzVector &b) const {
  return a+b;
}

KtLorentzVector KtRecomE::operator()(const KtLorentzVector &a) const {
  return a;
}

KtRecomPt::KtRecomPt() : m_name("Pt") {}
  //KtRecomPt::~KtRecomPt() {}
std::string KtRecomPt::name() const {return m_name;}

HepLorentzVector KtRecomPt::operator()(const HepLorentzVector &a, const HepLorentzVector &b) const {
  KtFloat pti, ptj, newPt, newEta, newPhi, deltaPhi;
  pti = a.perp();
  ptj = b.perp();
  newPt = pti + ptj;
  newEta = (pti * a.eta() + ptj * b.eta()) / newPt;
  deltaPhi = phiAngle(b.phi() - a.phi());
  newPhi = a.phi() + deltaPhi * ptj / newPt;
  newPhi = phiAngle(newPhi);
  return HepLorentzVector(newPt*cos(newPhi),newPt*sin(newPhi),newPt*sinh(newEta),newPt*cosh(newEta));
}

  /** Make 4-vector massless by setting E = p */
KtLorentzVector KtRecomPt::operator()(const KtLorentzVector &a) const {
  KtLorentzVector v = a;
  v.setE(a.vect().mag());
  return v;
}


KtRecomPt2::KtRecomPt2() : m_name("Pt^2") {}
  //KtRecomPt2::~KtRecomPt2() {}
std::string KtRecomPt2::name() const {return m_name;}

HepLorentzVector KtRecomPt2::operator()(const HepLorentzVector &a, const HepLorentzVector &b) const {
  KtFloat pti, ptj, ptisq, ptjsq, newPt, newEta, newPhi, deltaPhi;
  pti = a.perp();
  ptj = b.perp();
  ptisq = a.perp2();
  ptjsq = b.perp2();
  newPt = pti + ptj;
  newEta = (ptisq * a.eta() + ptjsq * b.eta()) / (ptisq + ptjsq);
  deltaPhi = phiAngle(b.phi() - a.phi());
  newPhi = a.phi() + deltaPhi * ptjsq / (ptisq + ptjsq);
  newPhi = phiAngle(newPhi);
  return HepLorentzVector(newPt*cos(newPhi),newPt*sin(newPhi),newPt*sinh(newEta),newPt*cosh(newEta));
}

  /** Make 4-vector massless by setting E = p */
KtLorentzVector KtRecomPt2::operator()(const KtLorentzVector &a) const {
  KtLorentzVector v = a;
  v.setE(a.vect().mag());
  return v;
}


KtRecomEt::KtRecomEt() : m_name("Et") {}
  //KtRecomEt::~KtRecomEt() {}
std::string KtRecomEt::name() const {return m_name;}

HepLorentzVector KtRecomEt::operator()(const HepLorentzVector &a, const HepLorentzVector &b) const {
  KtFloat pti, ptj, newPt, newEta, newPhi, deltaPhi;
  pti = a.et();
  ptj = b.et();
  newPt = pti + ptj;
  newEta = (pti * a.eta() + ptj * b.eta()) / newPt;
  deltaPhi = phiAngle(b.phi() - a.phi());
  newPhi = a.phi() + deltaPhi * ptj / newPt;
  newPhi = phiAngle(newPhi);
  return HepLorentzVector(newPt*cos(newPhi),newPt*sin(newPhi),newPt*sinh(newEta),newPt*cosh(newEta));
}

  /** Make 4-vector massless by scaling momentum to equal E */
KtLorentzVector KtRecomEt::operator()(const KtLorentzVector &a) const {
  KtLorentzVector v = a;
  KtFloat scale = a.e() / a.vect().mag();
  v.setVect(a.vect() * scale);
  return v;
}


KtRecomEt2::KtRecomEt2() : m_name("Et^2") {}
  //KtRecomEt2::~KtRecomEt2() {}
std::string KtRecomEt2::name() const {return m_name;}

HepLorentzVector KtRecomEt2::operator()(const HepLorentzVector &a, const HepLorentzVector &b) const {
  KtFloat pti, ptj, ptisq, ptjsq, newPt, newEta, newPhi, deltaPhi;
  pti = a.et();
  ptj = b.et();
  ptisq = pti*pti;
  ptjsq = ptj*ptj;
  newPt = pti + ptj;
  newEta = (ptisq * a.eta() + ptjsq * b.eta()) / (ptisq + ptjsq);
  deltaPhi = phiAngle(b.phi() - a.phi());
  newPhi = a.phi() + deltaPhi * ptjsq / (ptisq + ptjsq);
  newPhi = phiAngle(newPhi);
  return HepLorentzVector(newPt*cos(newPhi),newPt*sin(newPhi),newPt*sinh(newEta),newPt*cosh(newEta));
}

  /** Make 4-vector massless by scaling momentum to equal E */
KtLorentzVector KtRecomEt2::operator()(const KtLorentzVector &a) const {
  KtLorentzVector v = a;
  KtFloat scale = a.e() / a.vect().mag();
  v.setVect(a.vect() * scale);
  return v;
}

}//end of namespace
