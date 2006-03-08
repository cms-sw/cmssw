//  ProtJet2.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"

#include <vector>
#include <algorithm>               // include STL algorithm implementations
#include <numeric>		   // For the use of numeric

const double PI=3.14159265;

using std::vector;

ProtoJet::ProtoJet(): m_e(0), m_px(0), m_py(0), m_pz(0) {
}

ProtoJet::ProtoJet(const Candidates& theConstituents) {
  m_constituents = theConstituents;
  calculateLorentzVector(); 
}//end of constructor

ProtoJet::~ProtoJet() {
}

int ProtoJet::n90() const {
  vector<double> eList;
  for(Candidates::const_iterator i = m_constituents.begin(); i != m_constituents.end(); ++i) {
    eList.push_back((*i)->et());
  }
    
  //Make sure that we have a sorted list of constituents:  
  sort(eList.begin(), eList.end());
  
  int counter = 0;
  double e90 = et()*0.9;
  double etSum = 0.;
  int i = eList.size ();
  while (--i >=0) {
    etSum += eList[i];
    counter++;
    if (etSum >= e90) break;
  }
  return counter;
}

double ProtoJet::phi() const {
  double px_pos=fabs(m_px);
  double py_pos=fabs(m_py);
  double phi= 0.;

  phi = atan2 (py_pos, (px_pos + 1.e-20));
  if (phi < 0) phi += 2.*PI;
  return phi;
}


HepLorentzVector ProtoJet::getLorentzVector() const {
  HepLorentzVector theLorentzVector;

  theLorentzVector.setPx(px());
  theLorentzVector.setPy(py());
  theLorentzVector.setPz(pz());
  theLorentzVector.setE(e());

 return theLorentzVector;
}

void ProtoJet::calculateLorentzVector() {
  m_e = 0; m_px = 0; m_py = 0; m_pz = 0;
  for(Candidates::const_iterator i = m_constituents.begin(); i !=  m_constituents.end(); ++i) {
    const reco::Candidate* c = *i;
    m_e += c->energy();
    m_px += c->px();
    m_py += c->py();
    m_pz += c->pz();
  } //end of loop over the jet constituents
}



