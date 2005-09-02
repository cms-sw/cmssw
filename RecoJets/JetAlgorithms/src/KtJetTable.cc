#include "RecoJets/JetAlgorithms/interface/KtJetTable.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

namespace KtJet{

KtJetTable::KtJetTable(const std::vector<KtLorentzVector> & p, KtDistance *ktdist, KtRecom *recom)
  : m_fKtDist(ktdist), m_ktRecom(recom) {
  m_nRows = p.size();
  m_jets.reserve(m_nRows);          // Reserve space for jet array
  m_ddi.reserve(m_nRows);           // Reserve space for di array
  m_dPairs.resize(m_nRows);         // Make space for dij table

  std::vector<KtLorentzVector>::const_iterator pitr = p.begin();
  for (; pitr != p.end(); ++pitr) { // Initialize jets with one particle each
    KtLorentzVector j;              //   create new jet
    j.add(*pitr);                   //   add single particle to jet
    m_jets.push_back(j);
  }
  for (int i = 0; i < m_nRows-1; ++i) {     // Fill array of pair kt's
    for (int j = i+1 ; j < m_nRows; ++j) {
      KtFloat D = (*m_fKtDist)(m_jets[i],m_jets[j]);
      m_dPairs(i,j) = D;
    }
  }
  for (int i = 0; i < m_nRows; ++i){        // Fill vector of particle kt's
    m_ddi.push_back((*m_fKtDist)(m_jets[i]));
  }
}

KtJetTable::~KtJetTable() {}

const KtLorentzVector & KtJetTable::getJet(int i) const {
  return m_jets[i];
}

KtFloat KtJetTable::getD(int i, int j) const {
  return m_dPairs(i,j);
}

KtFloat KtJetTable::getD(int i) const {
  if (i<0 || i>=static_cast<int>(m_ddi.size())) {
    std::cout << "[Jets] ERROR in KtJetTable::getD(int)" << std::endl;
    std::cout << "[Jets]   i, m_ddi.size() = " << i << ", " << m_ddi.size() << std::endl;
  }
  return m_ddi[i];
}

/***************************************************************
 *  Merge jets i and j, updating four-momentum and kt vectors  *
 ***************************************************************/
void KtJetTable::mergeJets(int i, int j) {
  int njet = getNJets();
  if (i<0 || i>=njet || j<0 || j>=njet || i>=j) {
    std::cout << "[Jets] ERROR in KtJetTable::mergeJets" << std::endl;
    std::cout << "[Jets]   Attempt to merge jets " << i << ", " << j << " in event with " << njet << " jets" << std::endl;
  }
  m_jets[i].add(m_jets[j],m_ktRecom); // Add constituents and merge 4-momenta using required scheme
  for (int ii=0; ii<i; ++ii) {      // Calculate pair kt's involving merged particles
    KtFloat D = (*m_fKtDist)(m_jets[ii],m_jets[i]);
    m_dPairs(ii,i) = D;
  }
  for (int jj=i+1; jj<njet; ++jj) {
    KtFloat D = (*m_fKtDist)(m_jets[i],m_jets[jj]);
    m_dPairs(i,jj) = D;
  }
  m_ddi[i] = (*m_fKtDist)(m_jets[i]);  // Calculate kt of merged particles
  killJet(j);                       // Now delete particle j
}

/***************************************************
 *  Delete jet i by moving last jet on top of it,  *
 *   updating four momentum and kt vectors         *
 ***************************************************/
void KtJetTable::killJet(int i) {
  //  std::cout << " KtJetTable::killJet, i = " << i << std::endl;
  int njet = getNJets();
  if (i<0 || i>=njet) {
    std::cout << "[Jets] ERROR in KtEvent::killJet" << std::endl;
    std::cout << "[Jets]   Attempt to delete jet " << i << " in event with " << njet << " jets" << std::endl;
  }
  //  std::cout << " njet = " << njet << std::endl;
  m_jets[i] = m_jets[njet-1];      // move last jet into space left by jet i
  for (int j=0; j<i; ++j) {    // move pair kt's
    m_dPairs(j,i) = m_dPairs(j,njet-1);
  }
  for (int j=i+1; j<njet-1; ++j) {
    m_dPairs(i,j) = m_dPairs(j,njet-1);
  }
  m_ddi[i] = m_ddi[njet-1];          // move jet kt
  m_dPairs.killJet();           // reduce size of kt array
  m_jets.pop_back();             // delete last jet from vector
  m_ddi.pop_back();               // delete last jet kt
  //  std::cout << " Finished in KtJetTable::killJet" << std::endl;
}

std::pair<int,int> KtJetTable::getMinDPair() const {
  return m_dPairs.getMin();
}

int KtJetTable::getMinDJet() const {
  return std::distance(m_ddi.begin(),std::min_element(m_ddi.begin(),m_ddi.end())); 
}

/*************************************************************
 *  Now the functions for nested class KtJetTable::DijTable  *
 *************************************************************/

KtJetTable::DijTable::DijTable(int nParticles) : m_nRows(nParticles), m_nJets(nParticles) {
  m_table.resize(m_nRows*m_nRows);
}

KtJetTable::DijTable::~DijTable() {}

void KtJetTable::DijTable::resize(int nParticles) {
  /*************************************************************
   *  Reserve space for kt of pairs with nParticles particles  *
   *************************************************************/
  m_nRows = nParticles;
  m_nJets = nParticles;
  m_table.resize(m_nRows*m_nRows);
}

std::pair<int,int> KtJetTable::DijTable::getMin() const {
  /********************************************************
   *  Find position of smallest entry in table  *
   ********************************************************/
  KtFloat d = m_table[1];
  int k=0; int i=0; int j=1;                    // Initialize to first used element
  for (int ii=0; ii<m_nJets-1; ++ii) {
    for (int jj=ii+1; jj<m_nJets; ++jj) {
      ++k;
      if (m_table[k]<d) {
	i = ii; j = jj; d = m_table[k];
      }
    }
    k += 2 + ii + m_nRows - m_nJets;
  }
  return std::pair<int,int>(i,j);
}

void KtJetTable::DijTable::print() const {
  /*****************************************
   *  Write out contents of table to cout  *
   *****************************************/
  for (int i = 0; i<m_nRows-1; ++i){
    for (int j = i+1; j<m_nRows; ++j){
      KtFloat D = (*this)(i,j);
      std::cout << i+1 << " " << j+1 << " " << D << '\n';
    }
  }
  std::cout << std::endl;
}

KtFloat & KtJetTable::DijTable::operator()(int ii, int jj) {
  int i = std::min(ii,jj);
  int j = std::max(ii,jj);
  if (i<0 || j<0 || i>=m_nJets || j>=m_nJets || i>=j) {
    std::cout << "[Jets] ERROR in KtJetTable::DijTable::operator()" << std::endl;
    std::cout << "[Jets]   Attempt to access element (" << i << "," << j << ") in table with nJets, nRows = "
	 << m_nJets << ", " << m_nRows << std::endl;
  }
  return *(m_table.begin() + i*m_nRows + j);
}


KtFloat KtJetTable::DijTable::operator()(int i, int j) const {
  if (i<0 || j<0 || i>=m_nJets || j>=m_nJets) {
    std::cout << "[Jets] ERROR in KtJetTable::DijTable::operator() const";
    std::cout << "[Jets]   Attempt to access element (" << i << "," << j << ") in table with nJets, nRows = "
	 << m_nJets << ", " << m_nRows << std::endl;
  }
  return *(m_table.begin() + i*m_nRows + j);
}

void KtJetTable::DijTable::killJet() {
  if (m_nJets<=0) {
    std::cout << "[Jets] ERROR in KtJetTable::DijTable::killJet()" << std::endl;
    std::cout << "[Jets]   Called when m_nJets = " << m_nJets << std::endl;
  }
  --m_nJets;
}
}//end of namespace
