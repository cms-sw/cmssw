#ifndef JetAlgorithms_ExtKtJetAlgorithmWrapper_h
#define JetAlgorithms_ExtKtJetAlgorithmWrapper_h

/** \class ExtKtJetProducer
 *
 * ExtKtJetProducer is the EDProducer subclass which runs
 * the KtJet algorithm for jetfinding.
 * 
 * ktjet-package: (http://projects.hepforge.org/ktjet)
 * See Reference: Comp. Phys. Comm. vol 153/1 85-96 (2003)
 * Also:  http://www.arxiv.org/abs/hep-ph/0210022
 * this package is included in the external CMSSW-dependencies
 * License of package: GPL
 *
 * 
 * Producer by Andreas Oehler, Uni Karlsruhe
 * \version   1st Version Feb. 1 2007
 * 
 *
 ************************************************************/

#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class ExtKtJetAlgorithmWrapper
{
  typedef reco::CandidateRef FJCand;
  typedef const reco::CandidateRef constFJCand;
 public:
  ExtKtJetAlgorithmWrapper();
  ExtKtJetAlgorithmWrapper(const edm::ParameterSet& ps);
  ~ExtKtJetAlgorithmWrapper(){};
  void run (const std::vector <FJCand>& fInput, std::vector<ProtoJet>* fOutput) const;
 private:
  int theMode;
  double thePtMin;
  double theDcut;
  int theNjets;
  int theAngle;
  int theRecom;
  double theRparam;
  // theColType: 4 is pp
  int theColType;
};

#endif
