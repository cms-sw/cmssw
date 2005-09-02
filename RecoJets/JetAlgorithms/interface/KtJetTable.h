#ifndef JetAlgorithms_KtJetTable_h
#define JetAlgorithms_KtJetTable_h

#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#ifndef STD_VECTOR_H
#include <vector>
#define STD_VECTOR_H
#endif

#ifndef STD_UTILITY_H
#include <string>
#define STD_UTILITY_H
#endif


namespace KtJet{
  //class KtCalc;
  //  class KtLorentzVector;
/**
 *  Class KtJetTable encapsulates the jet four-momenta and Kt's in one object.
 *  Does merging and deletion of jets and returns jets, Kt's etc.
 *  Uses a KtDistance object to calculate Kt's.
 *
 *  Usage example:
 *    HepLorentzVector eScheme(const HepLorentzVector &, const HepLorentzVector &); // function to merge jets
 *    std::vector<KtLorentzVector> jets;
 *                                //  [Put some particles in "jets".]
 *    KtDistanceDeltaR ktPP(4);   //  4 = pp collision
 *    KtJetTable jt(jets, ktPP, eScheme);

 @author J.Butterworth J.Couchman B.Cox B.Waugh
 */
class KtJetTable {
 public:
  KtJetTable(const std::vector<KtLorentzVector> &, KtDistance *, KtRecom *recom);
  ~KtJetTable();
  /* Number of jets */
  inline int getNJets() const;                        
  /** Get jet from table */
  const KtLorentzVector & getJet(int i) const;    
  /** Kt for jet pair (i,j) */
  KtFloat getD(int i, int j) const;               
  /** Kt of jet (i) with respect to beam */
  KtFloat getD(int i) const;
  /** Get indices of jet pair with min Kt */
  std::pair<int,int> getMinDPair() const;
  /** Get index of jet with min Kt with respect to beam */
  int getMinDJet() const;                       
  /** Combine jets (i,j) (E-scheme only so far) */ 
  void mergeJets(int i, int j);                  
  /** Delete jet (i) from table */
  void killJet(int i);
 private:
  /** Initial number of jets/particles */
  int m_nRows;
  /** Jet 4-momenta */
  std::vector<KtLorentzVector> m_jets;
  /** Function object to define Kt distance scheme */
  KtDistance *m_fKtDist;
  /** Recombination scheme */
  KtRecom *m_ktRecom;
  //  HepLorentzVector (*m_frecom)(const HepLorentzVector &, const HepLorentzVector &);
  /** Kt with respect to beam */
  std::vector<KtFloat> m_ddi;
  /** Class to deal with pair Kt's */
  class DijTable {
  private:
    /** No. of initial jets/particles */
    int m_nRows;                 
    /** No. of jets after merging etc. */
    int m_nJets;
    /** Vector of Kt values */
    std::vector<KtFloat> m_table;
  public:
    DijTable(int nParticles=0);
    ~DijTable();
    /** Set size to hold nParticles particles */
    void resize(int nParticles);            
    /** Return reference to allow Kt value to be set */
    KtFloat & operator()(int i, int j);         
    /** Return Kt by value */
    KtFloat operator()(int i, int j) const;      
    /** Find position of minimum Kt in table */
    std::pair<int,int> getMin() const;           
    /** Decrement number of jets */
    void killJet();                             
    /** ??? debug only? Print contents of table */ 
    void print() const;                         
  };
  /** 2D table of all pair kt's */
  DijTable m_dPairs;
};

#include "RecoJets/JetAlgorithms/interface/KtJetTable.icc"

}//end of namespace
#endif
