#include "RecoJets/JetAlgorithms/interface/KtEvent.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "RecoJets/JetAlgorithms/interface/KtJetTable.h"
#include "RecoJets/JetAlgorithms/interface/KtDistance.h"
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#include "RecoJets/JetAlgorithms/interface/KtRecom.h"
#include "RecoJets/JetAlgorithms/interface/KtRecomInterface.h"
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>

namespace KtJet {

/** Constructor for inclusive method */
KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, int angle, int recom,
		 KtFloat rParameter)
  : m_type(type),
    m_rParameterSq(rParameter*rParameter), m_inclusive(true),
    m_jets() {
  init(constituents,0,angle,0,recom);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, KtDistance *angle, KtRecom *recom,
		 KtFloat rParameter)
  : m_type(type),
    m_rParameterSq(rParameter*rParameter), m_inclusive(true),
    m_jets() {
  init(constituents,angle,0,recom,0);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, KtDistance *angle, int recom,
		 KtFloat rParameter)
  : m_type(type),
    m_rParameterSq(rParameter*rParameter), m_inclusive(true),
    m_jets() {
  init(constituents,angle,0,0,recom);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, int angle, KtRecom *recom,
		 KtFloat rParameter)
  : m_type(type),
    m_rParameterSq(rParameter*rParameter), m_inclusive(true),
    m_jets() {
  init(constituents,0,angle,recom,0);
}

KtEvent::KtEvent(const std::vector<HepLorentzVector> & constituents,
                 int type, int angle, int recom,
		 KtFloat rParameter)
  : m_type(type),
    m_rParameterSq(rParameter*rParameter), m_inclusive(true),
    m_jets() {
  std::vector<KtLorentzVector> konstituents;
  makeKtFromHepLV(konstituents,constituents);
  init(konstituents,0,angle,0,recom);
}

/** Constructor for exclusive method */
KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
		 int type, int angle, int recom)
  : m_type(type),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(constituents,0,angle,0,recom);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, KtDistance *angle, KtRecom *recom)
  : m_type(type),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(constituents,angle,0,recom,0);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, KtDistance *angle, int recom)
  : m_type(type),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(constituents,angle,0,0,recom);
}

KtEvent::KtEvent(const std::vector<KtLorentzVector> & constituents,
                 int type, int angle, KtRecom *recom)
  : m_type(type),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(constituents,0,angle,recom,0);
}

KtEvent::KtEvent(const std::vector<HepLorentzVector> & constituents,
                 int type, int angle, int recom)
  : m_type(type),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  std::vector<KtLorentzVector> konstituents;
  makeKtFromHepLV(konstituents,constituents);
  init(konstituents,0,angle,0,recom);
}

/** Constructor for (exclusive) subjet method */
KtEvent::KtEvent(const KtLorentzVector & jet, int angle, int recom)
  : m_type(1),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(jet,0,angle,0,recom);
}

KtEvent::KtEvent(const KtLorentzVector & jet, KtDistance *angle, KtRecom *recom)
  : m_type(1),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(jet,angle,0,recom,0);
}

KtEvent::KtEvent(const KtLorentzVector & jet, KtDistance *angle, int recom)
  : m_type(1),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(jet,angle,0,0,recom);
}

KtEvent::KtEvent(const KtLorentzVector & jet, int angle, KtRecom *recom)
  : m_type(1),
    m_rParameterSq(1), m_inclusive(false),
    m_jets() {
  init(jet,0,angle,recom,0);
}

KtEvent::~KtEvent() {
  delete m_ktDist;
  delete m_ktRecom;
}

void KtEvent::findJetsN(int nJets) {
  // Uses merging history created by makeJets
  if (m_inclusive) {
    std::cout << "[Jets] WARNING in KtEvent::findJetsN : Trying to find exclusive jets in inclusive event\n";
  }
  m_jets.clear();
  KtJetTable jt(m_constituents,m_ktDist,m_ktRecom);
  int nParticles = m_constituents.size();
  int njet = m_constituents.size();
  while (njet>nJets) {
    int hist = m_hist[njet-1];
    if (hist>=0) {
      jt.killJet(hist);
    } else {
      // Merge two jets
      hist = -hist;
      int iPair = hist % nParticles;
      int jPair = hist / nParticles;
      jt.mergeJets(iPair,jPair);
    }
    --njet;
  }
  // Now jets remaining in jt should be required jets: copy to m_jets
  for (int i=0; i<jt.getNJets(); ++i) { m_jets.push_back(jt.getJet(i)); }
}

/** Do exclusive jet-finding to scale dCut */
void KtEvent::findJetsD(KtFloat dCut) {
  // Find number of jets at scale dCut, then call findJetsN to do merging
  int njets = 1;
  for (int iloop = m_constituents.size(); iloop>1; --iloop) {
    if (m_dMerge[iloop-1]>=dCut) {
      njets = iloop;
	break;
    }
  }
  findJetsN(njets);
}

/** Do exclusive jet-finding to scale yCut */
void KtEvent::findJetsY(KtFloat yCut) {
  // Find jets at scale yCut by rescaling to give dCut, then calling findJetsD
  KtFloat dCut = yCut * m_eCut * m_eCut;
  findJetsD(dCut);
}

/** Get jets */
std::vector<KtLorentzVector> KtEvent::getJets() {
  return m_jets;
}

std::vector<KtLorentzVector> KtEvent::getJetsE() {
  std::vector<KtLorentzVector> a(m_jets);
  std::sort(a.begin(),a.end(),greaterE);
  return a;
}

std::vector<KtLorentzVector> KtEvent::getJetsEt() {
  std::vector<KtLorentzVector> a(m_jets);
  std::sort(a.begin(),a.end(),greaterEt);
  return a;
}

std::vector<KtLorentzVector> KtEvent::getJetsPt() {
  std::vector<KtLorentzVector> a(m_jets);
  std::sort(a.begin(),a.end(),greaterPt);
  return a;
}

std::vector<KtLorentzVector> KtEvent::getJetsRapidity() {
  std::vector<KtLorentzVector> a(m_jets);
  std::sort(a.begin(),a.end(),greaterRapidity);
  return a;
}

std::vector<KtLorentzVector> KtEvent::getJetsEta() {
  std::vector<KtLorentzVector> a(m_jets);
  std::sort(a.begin(),a.end(),greaterEta);
  return a;
}

std::vector<const KtLorentzVector *> KtEvent::getConstituents() const {
  std::vector<const KtLorentzVector *> a;
  std::vector<KtLorentzVector>::const_iterator itr = m_constituents.begin();
  for (; itr != m_constituents.end(); ++itr) {
    a.push_back(&*itr);   // ??? Converted to pointer?
  }
  return a;
}

std::vector<KtLorentzVector> KtEvent::copyConstituents() const {
  std::vector<KtLorentzVector> a(m_constituents);
  return a;
}

KtLorentzVector KtEvent::getJet(const KtLorentzVector & a) const {
  std::vector<KtLorentzVector>::const_iterator itr = m_jets.begin();
  for (; itr != m_jets.end(); ++itr) {
    if (a == *itr || itr->contains(a)) return *itr;
  }
  std::cout << "[Jets] ERROR in KtEvent::getJet : particle not in event" << std::endl;
  return a; // ??? Do something more sensible here?
}


  /*********************
   *  Private methods  *
   *********************/

  /** Initialization for event jet analysis */
void KtEvent::init(const std::vector<KtLorentzVector> & constituents,
		   KtDistance* dist, int idist, KtRecom* recom, int irecom) {
  static bool first_inclusive = true, first_exclusive = true;
  listAuthors();
  setSchemes(dist,idist,recom,irecom);
  m_constituents.clear();
  std::vector<KtLorentzVector>::const_iterator itr;
  for (itr=constituents.begin(); itr!=constituents.end(); ++itr) {
    KtLorentzVector v = (*m_ktRecom)(*itr);
    m_constituents.push_back(v);
  }
  addEnergy();
  setECut(getETot());
  if (first_inclusive && m_inclusive) {
    first_inclusive = false;
    printSteering("inclusive");
  }
  if (first_exclusive && !m_inclusive) {
    first_exclusive = false;
    printSteering("exclusive");
  }
  makeJets();
  if (!isInclusive()) m_jets.clear(); // Only need merging history in exclusive case.
                                      // Jets reconstructed later with fixed d cut or njets.
}

  /** Initialization for subjet analysis */
void KtEvent::init(const KtLorentzVector & jet, KtDistance* dist, int idist, KtRecom* recom, int irecom) {
  static bool first_subjet = true;
  listAuthors();
  setSchemes(dist,idist,recom,irecom);
  std::vector<const KtLorentzVector*> constituents = jet.getConstituents();
  m_constituents.clear();
  std::vector<const KtLorentzVector *>::const_iterator itr;
  for (itr=constituents.begin(); itr!=constituents.end(); ++itr) {
    KtLorentzVector v = (*m_ktRecom)(**itr);
    m_constituents.push_back(v);
  }
  addEnergy();
  setECut(jet.perp());
  if (first_subjet) {
    first_subjet = false;
    printSteering("subjet");
  }
  makeJets();
  m_jets.clear();  // Only need merging history. Jets reconstructed later with fixed d cut or njets.
}

void KtEvent::addEnergy() {
  m_eTot = 0;
  std::vector<KtLorentzVector>::const_iterator itr;
  for (itr = m_constituents.begin(); itr != m_constituents.end(); ++itr) {
    m_eTot += itr->e();          // Add up energy in event
  }
}

void KtEvent::makeJets() {
  int nParticles = m_constituents.size();
  m_dMerge.resize(nParticles);              // Reserve space for D-cut vector
  m_hist.resize(nParticles);                // Reserve space for merging history vector
  KtJetTable jt(m_constituents,m_ktDist,m_ktRecom);

  int njet;
  //  KtFloat dMax = 0;
  while ((njet=jt.getNJets())>1) {          // Keep merging until only one jet left
    int iPairMin, jPairMin, iJetMin;
    KtFloat dPairMin, dJetMin, dMin;
    std::pair<int,int> pPairMin = jt.getMinDPair(); // Find jet pair with minimum D
    iPairMin = pPairMin.first;
    jPairMin = pPairMin.second;
    dPairMin = jt.getD(iPairMin,jPairMin);
    iJetMin = jt.getMinDJet();                      // Find jet with minimum D to beam
    dJetMin = jt.getD(iJetMin);
    dJetMin *= m_rParameterSq;                      // Scale D to beam by rParameter
    bool mergeWithBeam = ((m_type != 1 && (dJetMin<=dPairMin)) || njet==1); // Merge with beam remnant?
    dMin = mergeWithBeam ? dJetMin : dPairMin;

    if (mergeWithBeam) {                        // Merge jet with beam remnant
      m_jets.push_back(jt.getJet(iJetMin));     //   Add jet to inclusive list
      m_hist[njet-1] = iJetMin;                 //   Record which jet killed (+ve to indicate merge with beam)
      jt.killJet(iJetMin);                      //   Chuck out jet
    } else {                                    // Merge pair of jets
      m_hist[njet-1] = -(jPairMin*nParticles+iPairMin); //  Record which jet pair merged (-ve to indicate pair)
      jt.mergeJets(iPairMin,jPairMin);                  //  Merge jets
    }
    m_dMerge[njet-1] = dMin;                    // Store D where jets merged
  }
  // End of loop: now should have njet = 1
  m_jets.push_back(jt.getJet(0));                           // Add last jet to list
  m_dMerge[0] = jt.getD(0);                                    // Put last jet kt in vectors ???
}

void KtEvent::makeKtFromHepLV(std::vector<KtLorentzVector> & kt,
			      const std::vector<HepLorentzVector> & hep) {
  std::vector<HepLorentzVector>::const_iterator itr = hep.begin();
  for (; itr != hep.end(); ++itr) {
    KtLorentzVector ktvec(*itr);
    kt.push_back(ktvec);
  }
}

void KtEvent::setSchemes(KtDistance* dist, int idist, KtRecom* recom, int irecom) {
  m_angle = idist;
  m_ktDist = (dist) ? dist : getDistanceScheme(idist,m_type);
  m_recom = irecom;
  m_ktRecom = (recom) ? recom : getRecomScheme(irecom);
}

void KtEvent::printSteering(std::string mode) const {
  static std::string collisionType[4] = {"e+e-","ep","pe","pp"};
  std::string type = (m_type>=1 && m_type <= 4)  ? collisionType[m_type-1]  : "unknown";
  std::string angle = m_ktDist->name();
  std::string recom = m_ktRecom->name();
  mode.resize(10,' ');
  type.resize(16,' ');
  angle.resize(16,' ');
  recom.resize(16,' ');
  std::cout << "[Jets] ******************************************\n";
  std::cout << "[Jets] * KtEvent constructor called: " << mode << " *\n";
  std::cout << "[Jets] * Collision type:       " << type  << " *\n";
  std::cout << "[Jets] * Kt scheme:            " << angle << " *\n";
  std::cout << "[Jets] * Recombination scheme: " << recom << " *\n";
  //#ifdef KTDOUBLEPRECISION
  std::cout << "[Jets] * Compiled to use double precision.      *\n";
  //#else
  //std::cout << "* Compiled to use single precision.      *\n";
  //#endif
  std::cout << "[Jets] ******************************************\n";
}

void KtEvent::listAuthors() const {
  static bool first = true;
  if (first) {
    first = false;
    std::cout << "[Jets] ***********************************************\n";
    std::cout << "[Jets] * Package KtJet written by:                   *\n";
    std::cout << "[Jets] *   Jon Butterworth                           *\n";
    std::cout << "[Jets] *   Jon Couchman                              *\n";
    std::cout << "[Jets] *   Brian Cox                                 *\n";
    std::cout << "[Jets] *   Ben Waugh                                 *\n";
    std::cout << "[Jets] * See documentation at <http://www.ktjet.org> *\n";
    std::cout << "[Jets] *                                             *\n";
    std::cout << "[Jets] * Version:                                    *\n";
    std::cout << "[Jets] *   v1.0 (2005-22-04)                         *\n";
    std::cout << "[Jets] * CMS EDM adaption:                           *\n";
    std::cout << "[Jets] *   Fernando Varela (Boston University)       *\n";
    std::cout << "[Jets] * send comments, bug reports to:              *\n";
    std::cout << "[Jets] *   fvarela@bu.edu                            *\n";
    std::cout << "[Jets] ***********************************************\n";
  }
}

}//end of namespace
