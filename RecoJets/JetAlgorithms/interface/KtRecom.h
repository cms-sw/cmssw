#ifndef JetAlgorithms_KtRecom_h
#define JetAlgorithms_KtRecom_h

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

#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"

namespace KtJet {
/**
 *  Function object to combine 4-momenta
 
 @author J.Butterworth J.Couchman B.Cox B.Waugh
*/

/** Get required KtRecom object given integer argument
 */
KtRecom* getRecomScheme(int recom);

class KtRecomE : public KtRecom {
 public:
  KtRecomE();
  virtual ~KtRecomE(){};
  /** Return merged 4-momentum */
  HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const;
  /** Process input 4-momentum */
  KtLorentzVector operator()(const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  std::string m_name;
};


class KtRecomPt : public KtRecom {
 public:
  KtRecomPt();
  virtual ~KtRecomPt(){};
  /** Return merged 4-momentum */
  HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const;
  /** Process input 4-momentum */
  KtLorentzVector operator()(const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  std::string m_name;
};


class KtRecomPt2 : public KtRecom {
 public:
  KtRecomPt2();
  virtual ~KtRecomPt2(){};
  /** Return merged 4-momentum */
  HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const;
  /** Process input 4-momentum */
  KtLorentzVector operator()(const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  std::string m_name;
};


class KtRecomEt : public KtRecom {
 public:
  KtRecomEt();
  virtual ~KtRecomEt(){};
  /** Return merged 4-momentum */
  HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const;
  /** Process input 4-momentum */
  KtLorentzVector operator()(const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  std::string m_name;
};


class KtRecomEt2 : public KtRecom {
 public:
  KtRecomEt2();
  virtual ~KtRecomEt2(){};
  /** Return merged 4-momentum */
  HepLorentzVector operator()(const HepLorentzVector &, const HepLorentzVector &) const;
  /** Process input 4-momentum */
  KtLorentzVector operator()(const KtLorentzVector &) const;
  /** Name of scheme */
  std::string name() const;
 private:
  std::string m_name;
};

}//end of namespace

#endif
