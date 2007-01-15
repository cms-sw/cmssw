#ifndef JetAlgorithms_FastJetAlgorithmWrapper_h
#define JetAlgorithms_FastJetAlgorithmWrapper_h

/** \class FastJetAlgorithmWrapper
 *
 * FastJetAlgorithmWrapper is the Wrapper subclass which runs
 * the FastJetAlgorithm for jetfinding. 
 * 
 * The FastJet package, written by Matteo Cacciari and Gavin Salam, 
 * provides a fast implementation of the longitudinally invariant kt 
 * and longitudinally invariant inclusive Cambridge/Aachen jet finders.
 * More information can be found at:
 * http://parthe.lpthe.jussieu.fr/~salam/fastjet/
 *
 * \authors Andreas Oehler, University Karlsruhe (TH)
 * and Dorian Kcira, Institut de Physique Nucleaire
 * Departement de Physique
 * Universite Catholique de Louvain
 * have written the FastJetAlgorithmWrapper class
 * which uses the above mentioned package within the Framework
 * of CMSSW
 *
 * \version   1st Version Nov. 6 2006
 * 
 *
 *  
 *
 ************************************************************/

#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class FastJetAlgorithmWrapper
{
  typedef reco::CandidateRef FJCand;
  typedef const reco::CandidateRef constFJCand;
 public:
  FastJetAlgorithmWrapper();
  FastJetAlgorithmWrapper(const edm::ParameterSet& ps);
  ~FastJetAlgorithmWrapper();
  void run (const std::vector <FJCand>& fInput, std::vector<ProtoJet>* fOutput);
 private:
  //fastjet::JetDefinition jet_def;
  struct JetConfig;
  int theMode;
  JetConfig *theJetConfig;
  double thePtMin;
  double theDcut;
  double theInputMinE;
  int theNjets;
  // Subtraction-Parameters:
  bool theDoSubtraction;
  double theGhost_EtaMax;
  int theActive_Area_Repeats;
  //fastjet::ActiveAreaSpec theArea_Spec;
  double theGhostArea;
  double theMedian_Pt_Per_Area;
};


#endif
