#ifndef KTJET_KTEVENT_H
#define KTJET_KTEVENT_H

#ifndef KTJET_KTUTIL_H
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#define KTJET_KTUTIL_H
#endif

#ifndef KTJET_KTDISTANCEINTERFACE_H
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#define KTJET_KTDISTANCEINTERFACE_H
#endif

#ifndef KTJET_KTJETTABLE_H
#include "RecoJets/JetAlgorithms/interface/KtJetTable.h"
#define KTJET_KTJETTABLE_H
#endif

#ifndef KTJET_KTRECOMINTERFACE_H
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#define KTJET_KTRECOMINTERFACE_H
#endif

#ifndef STD_VECTOR_H
#include <vector>
#define STD_VECTOR_H
#endif

#ifndef STD_STRING_H
#include <string>
#define STD_STRING_H
#endif

#ifndef CLHEP_CLHEP_H
#include "CLHEP/config/CLHEP.h"
#define CLHEP_CLHEP_H
#endif

#ifndef CLHEP_LORENTZVECTOR_H
#include "CLHEP/Vector/LorentzVector.h"
#define CLHEP_LORENTZVECTOR_H
#endif


namespace KtJet {

class KtLorentzVector;

/**
 * The KtEvent class represents a whole system
 * of KtLorentzVectors constructed using 
 * the defined KT clustering algorithm.
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/

class KtEvent {
public:
  /** Inclusive method constructors */
  KtEvent(const std::vector<KtLorentzVector> &, int type, int angle, int recom,
	  KtFloat rparameter);
  KtEvent(const std::vector<KtLorentzVector> &, int type, KtDistance *, KtRecom *,
	  KtFloat rparameter);
  KtEvent(const std::vector<KtLorentzVector> &, int type, KtDistance *, int recom,
	  KtFloat rparameter);
  KtEvent(const std::vector<KtLorentzVector> &, int type, int angle, KtRecom *,
	  KtFloat rparameter);
  KtEvent(const std::vector<HepLorentzVector> &, int type, int angle, int recom,
	  KtFloat rparameter);
  /** Exclusive method constructors */
  KtEvent(const std::vector<KtLorentzVector> &, int type, int angle, int recom);
  KtEvent(const std::vector<KtLorentzVector> &, int type, KtDistance *, KtRecom *);
  KtEvent(const std::vector<KtLorentzVector> &, int type, KtDistance *, int recom);
  KtEvent(const std::vector<KtLorentzVector> &, int type, int angle, KtRecom *);
  KtEvent(const std::vector<HepLorentzVector> &, int type, int angle, int recom);
  /** Subjets method constructors */
  KtEvent(const KtLorentzVector & jet, int angle, int recom);
  KtEvent(const KtLorentzVector & jet, KtDistance *, KtRecom *);
  KtEvent(const KtLorentzVector & jet, KtDistance *, int recom);
  KtEvent(const KtLorentzVector & jet, int angle, KtRecom *);
  /** Destructor */
  ~KtEvent();

  /** Do exclusive jet-finding for nJets jets */
  void findJetsN(int nJets);
  /** Do exclusive jet-finding up to scale dCut */
  void findJetsD(KtFloat dCut);
  /** Do exclusive jet-finding up to parameter yCut */
  void findJetsY(KtFloat yCut);

  /** Returns the number of final state jets */
  inline int getNJets() const;
  /** Return final state jets without sorting */
  std::vector<KtLorentzVector> getJets();
  /** Return jets in order of decreasing E */
  std::vector<KtLorentzVector> getJetsE();
  /** Return final state jets in order of decreasing Et */
  std::vector<KtLorentzVector> getJetsEt();
  /** Return final state jets in order of decreasing Pt */
  std::vector<KtLorentzVector> getJetsPt();
  /** Return final state jets in order of decreasing rapidity */
  std::vector<KtLorentzVector> getJetsRapidity();
  /** Return final state jets in order of decreasing pseudorapidity (eta) */
  std::vector<KtLorentzVector> getJetsEta();

  /** d-cut where n+1 jets merged to n */
  inline KtFloat getDMerge(int nJets) const;
  /** y-cut where n+1 jets merged to n */
  inline KtFloat getYMerge(int nJets) const;

  /** Get number of objects input to KtEvent */
  inline int getNConstituents() const;
  /** Get pointers to input particles */
  std::vector<const KtLorentzVector *> getConstituents() const;
  /** Get copies of input particles */
  std::vector<KtLorentzVector> copyConstituents() const;

 /** Jet containing given particle
   *  If passed jet in this event, return same jet. If passed particle or jet not in this
   *  event, error. */
  KtLorentzVector getJet(const KtLorentzVector &) const;

  /** Set ECut value used in calculating YCut. Default is total transverse energy of the event */
  inline void setECut(KtFloat eCut); 
  /** Get ECut value used in calculating YCut */
  inline KtFloat getECut() const;
  /** Get total energy in event */
  inline KtFloat getETot() const;
  /** Get collision type */
  inline int getType() const;
  /** Get distance ("angle") scheme */
  inline int getAngle() const;
  /** Get recombination scheme */
  inline int getRecom() const;
  /** Get inclusive flag: true if inclusive method constructor was used */
  inline bool isInclusive() const;

private:
 
  /** Copy of original input particles */
  std::vector<KtLorentzVector> m_constituents;
  /** Collision type */
  int m_type;
  /** Kt distance scheme */
  int m_angle;
  /** Recombination scheme */
  int m_recom;
  KtRecom *m_ktRecom;
  /** R parameter squared */
  KtFloat m_rParameterSq;
  /** Flag for inclusive jets (false = exclusive) */
  bool m_inclusive;
  /** Energy scale for calculating y */
  KtFloat m_eCut;
  /** 1 / (eCut^2) */
  KtFloat m_etsq;
  /** Total energy in event */
  KtFloat m_eTot;
  /** d at each merge */
  std::vector<KtFloat> m_dMerge;
  /** Jets found */
  std::vector<KtLorentzVector> m_jets;
  /** merging history - jet/pair merged at each step */
  std::vector<int> m_hist;
  /** Function object to calculate jet resolution parameters */
  KtDistance *m_ktDist;

  /** Initialize event jet analysis */
  void init(const std::vector<KtLorentzVector> &, KtDistance* dist, int idist, KtRecom* recom, int irecom);
  /** Initialize subjet analysis */
  void init(const KtLorentzVector & jet, KtDistance* dist, int idist, KtRecom* recom, int irecom);
  /** Add up E and Et in event */
  void addEnergy();
  /** Make jets */
  void makeJets();
  /** Make KtLorentzVectors from HepLorentzVectors */
  void makeKtFromHepLV(std::vector<KtLorentzVector> &, const std::vector<HepLorentzVector> &);
  /** Get right distance and recombinatoin schemes based on integer flags as necessary */
  void setSchemes(KtDistance* dist, int idist, KtRecom* recom, int irecom);
  /** Print steering parameters to stdout */
  void printSteering(std::string mode="") const;
  /** Print list of authors, URL of documentation etc. */
  void listAuthors() const;
};

#include "RecoJets/JetAlgorithms/interface/KtEvent.icc"

}//end of namespace

#endif
