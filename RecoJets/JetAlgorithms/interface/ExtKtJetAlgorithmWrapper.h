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

#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class ExtKtJetAlgorithmWrapper
{
 public:
  ExtKtJetAlgorithmWrapper();
  ExtKtJetAlgorithmWrapper(const edm::ParameterSet& ps);
  ~ExtKtJetAlgorithmWrapper(){};
  void run (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) const;
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
