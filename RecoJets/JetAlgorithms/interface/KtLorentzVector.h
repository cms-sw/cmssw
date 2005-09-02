#ifndef JetAlgorithms_KtLorentzVector_h
#define JetAlgorithms_KtLorentzVector_h


#ifndef STD_VECTOR_H
#include <vector>
#define STD_VECTOR_H
#endif
#ifndef STD_CMATH_H
#include <cmath>
#define STD_CMATH_H
#endif
#ifndef CLHEP_CLHEP_H
#include "CLHEP/config/CLHEP.h"
#define CLHEP_CLHEP_H
#endif
#ifndef CLHEP_THREEVECTOR_H
#include "CLHEP/Vector/ThreeVector.h"
#define CLHEP_THREEVECTOR_H
#endif
#ifndef CLHEP_LORENTZVECTOR_H
#include "CLHEP/Vector/LorentzVector.h"
#define CLHEP_LORENTZVECTOR_H
#endif

#include "RecoJets/JetAlgorithms/interface/KtUtil.h"

namespace KtJet {
/**
 \class KtLorentzVector
 \brief
 * This Class represents a KtCluster Object.
 * It is just a CLHEP HepLorentzVector and also contains
 * a std::vector of all its constituent KtLorentzVectors 
 * and an id number
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/
class KtRecom;

class KtLorentzVector : public HepLorentzVector {
public:
  /** Default Constructor: create jet with no constituents */
  KtLorentzVector();
  /** Constructor: create particle with given 4-momentum */
  KtLorentzVector(const HepLorentzVector &);
  /** Constructor: create particle with given 4-momentum */
  KtLorentzVector(KtFloat px, KtFloat py, KtFloat pz, KtFloat e);

  /** Destructor */
  ~KtLorentzVector();

  /** return a reference to the vector of pointers of the KtLorentzVectors constituents */ 
  inline const std::vector<const KtLorentzVector*> & getConstituents() const;
  /** copy constituents */
  std::vector<KtLorentzVector> copyConstituents() const;
  /** returns the number of constituents KtLorentzVector is made up of */
  inline int getNConstituents() const;
  /** Check if a KtLorentzVector is a constituent */
  bool contains(const KtLorentzVector &) const;
  /** Add particle to jet using required recombination scheme to merge 4-momenta */
  void add(const KtLorentzVector &, KtRecom *recom);
  /** Add particle to jet using E scheme (4-vector addition) to merge 4-momenta */
  void add(const KtLorentzVector &);
  inline unsigned int getID() const {return m_id;}     // ??? Temporary, for debugging only
  /**  is it a Jet, not single particle */
  inline bool isJet() const;
  /** Add particle to jet using E scheme (4-vector addition) to merge 4-momenta */
  KtLorentzVector & operator+=(const KtLorentzVector &);
  /** Compare IDs of objects */
  inline bool operator==(const KtLorentzVector &) const;
  inline bool operator!=(const KtLorentzVector &) const;
  inline bool operator<(const KtLorentzVector &) const;
  inline bool operator>(const KtLorentzVector &) const;
private:
  /** rapidity, only valid if haven't called other methods since last call to add() */
  inline KtFloat crapidity() const;
  /** rapidity, only valid if haven't called other methods since last call to add() */
  KtFloat m_crapidity;
  /** calculate rapidity */
  inline void calcRapidity();
  /** private method to help add constituents to vector */
  void addConstituents(const KtLorentzVector*) ; 
  /** KtLorentzVectors id number */
  unsigned int m_id;
  /** Pointers to constituents */
  std::vector<const KtLorentzVector*> m_constituents;    
  /** Particle rather than jet */
  bool m_isAtomic;
  /** Number of instances so far */
  static unsigned int m_num;
  /** Some classes need access to crapidity for efficiency reasons */
  friend class KtDistanceDeltaR;
  friend class KtDistanceQCD;
};

bool greaterE(const HepLorentzVector &, const HepLorentzVector &);
bool greaterEt(const HepLorentzVector &, const HepLorentzVector &);
bool greaterPt(const HepLorentzVector &, const HepLorentzVector &);
bool greaterRapidity(const HepLorentzVector &, const HepLorentzVector &);
bool greaterEta(const HepLorentzVector &, const HepLorentzVector &);

#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.icc"

}//end of namespace
#endif
