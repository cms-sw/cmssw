#ifndef KTJET_KTDISTANCE_H
#define KTJET_KTDISTANCE_H

#ifndef KTJET_KTDISTANCEINTERFACE_H
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#define KTJET_KTDISTANCEINTERFACE_H
#endif

#ifndef STD_STRING_H
#include <string>
#define STD_STRING_H
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

#ifndef KTJET_KTUTIL_H
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#define KTJET_KTUTIL_H
#endif

#ifndef KTJET_KTLORENTZVECTOR_H
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#define KTJET_KTLORENTZVECTOR_H
#endif



namespace KtJet {
/**
 *  Function object to calculate Kt for jets and pairs.  
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/

/** Get required KtDistance object given integer argument
 */
KtDistance* getDistanceScheme(int dist, int collision_type);

class KtDistanceAngle : public KtDistance {
 public:
  KtDistanceAngle(int collision_type=1);
  virtual ~KtDistanceAngle(){}
  /** Jet Kt */
  KtFloat operator()(const KtLorentzVector &) const;
  /** Pair Kt */
  KtFloat operator()(const KtLorentzVector &, const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  int m_type;
  std::string m_name;
};


class KtDistanceDeltaR : public KtDistance {
 public:
  KtDistanceDeltaR(int collision_type=1);
  virtual ~KtDistanceDeltaR(){};
  /** Jet Kt */
  KtFloat operator()(const KtLorentzVector &) const;
  /** Pair Kt */
  KtFloat operator()(const KtLorentzVector &, const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  int m_type;
  std::string m_name;
};


class KtDistanceQCD : public KtDistance {
 public:
  KtDistanceQCD(int collision_type=1);
  virtual ~KtDistanceQCD(){};
  /** Jet Kt */
  KtFloat operator()(const KtLorentzVector &) const;
  /** Pair Kt */
  KtFloat operator()(const KtLorentzVector &, const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  int m_type;
  std::string m_name;
};

}//end of namespace

#endif
