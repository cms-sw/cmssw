#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#include <iostream>
#include <climits>
namespace KtJet {
unsigned int KtLorentzVector::m_num = 0;

/*************************************************
 *  Default constructor, used to create new jet  *
 *************************************************/
KtLorentzVector::KtLorentzVector() :
  HepLorentzVector(), m_constituents(), m_isAtomic(false) {
  if (m_num < UINT_MAX) {
    ++m_num;
  } else {
    m_num = 0;
    std::cout << "[Jets] Warning: Number of KtLorentzVectors exceeds capacity of unsigned int" << std::endl;
  }
  m_id = m_num;
}

/*****************************************************************
 *  Constructor for new "atomic" particle from HepLorentzVector  *
 *****************************************************************/
KtLorentzVector::KtLorentzVector(const HepLorentzVector &p) : 
  HepLorentzVector(p), m_id(m_num++), m_constituents(), m_isAtomic(true) {
}

/***********************************************************
 *  Constructor for new "atomic" particle from 4-momentum  *
 ***********************************************************/
KtLorentzVector::KtLorentzVector(KtFloat px, KtFloat py, KtFloat pz, KtFloat e) : 
  HepLorentzVector(px,py,pz,e), m_id(m_num++), m_constituents(), m_isAtomic(true) {
}

/**********************************************************/

KtLorentzVector::~KtLorentzVector() {}

/**********************************************************/

std::vector<KtLorentzVector> KtLorentzVector::copyConstituents() const {
  std::vector<KtLorentzVector> a;
  std::vector<const KtLorentzVector*>::const_iterator itr = m_constituents.begin();
  for (; itr != m_constituents.end(); ++itr) {
    a.push_back(**itr);
  }
  return a;
}

void KtLorentzVector::addConstituents(const KtLorentzVector* ktvec) {
  if(!ktvec->isJet()) {
    m_constituents.push_back((ktvec));
    return;
  }else{
    std::vector<const KtLorentzVector*>::const_iterator itr = ktvec->getConstituents().begin();
    for (; itr != ktvec->getConstituents().end() ; ++itr) this->addConstituents(*itr);
    return;
  }
}

bool KtLorentzVector::contains(const KtLorentzVector & a) const {
  if (a == *this) return true;
  if (m_isAtomic) return false;
  std::vector<const KtLorentzVector*>::const_iterator itr = m_constituents.begin();
  for (; itr != m_constituents.end() ; ++itr) {
    if (a == **itr) return true;
  }
  return false;
}

void KtLorentzVector::add(const KtLorentzVector &p, KtRecom *recom) {
  if (m_isAtomic) { // ???
    std::cout << "[Jets] Tried to add to atomic KtLorentzVector. You shouldn't do that!" << std::endl;
    //    exit(1);
  }
  this->addConstituents(&p);
  HepLorentzVector::operator=((*recom)((*this),p));
  calcRapidity();
}

void KtLorentzVector::add(const KtLorentzVector &p) {
  this->operator+=(p);
}

KtLorentzVector & KtLorentzVector::operator+= (const KtLorentzVector &p){
  if (m_isAtomic) { // ???
    std::cout << "[Jets] Tried to add to atomic KtLorentzVector. You shouldn't do that!" << std::endl;
    //    exit(1);
  }
  this->addConstituents(&p);
  HepLorentzVector::operator+=(p);
  calcRapidity();
  return *this;
}

// KtLorentzVector::KtLorentzVector(const KtLorentzVector & p) {} // Use default copy

/**************************************
 *  Comparison functions for sorting  *
 **************************************/
bool greaterE(const HepLorentzVector & a, const HepLorentzVector & b) {
  return (a.e()>b.e());
}

bool greaterEt(const HepLorentzVector & a, const HepLorentzVector & b) {
  return (a.et()>b.et());
}

bool greaterPt(const HepLorentzVector & a, const HepLorentzVector & b) {
  return (a.perp2()>b.perp2());
}

bool greaterRapidity(const HepLorentzVector & a, const HepLorentzVector & b) {
  return (a.rapidity()>b.rapidity());
}

bool greaterEta(const HepLorentzVector & a, const HepLorentzVector & b) {
  return (a.pseudoRapidity()>b.pseudoRapidity());
}

}//end of namespace
