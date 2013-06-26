
#ifndef JetAlgorithms_FastJetFWLiteWrapper_h
#define JetAlgorithms_FastJetFWLiteWrapper_h

/** \class FastJetFWLiteWrapper
 *
 * FastJetFWLiteWrapper is the Wrapper subclass which runs
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
 * have written the FastJetFWLiteWrapper class
 * which uses the above mentioned package within the Framework
 * of CMSSW
 *
 * \version   1st Version Nov. 6 2006
 * 
 *
 *  
 *
 ************************************************************/

#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"
#include "fastjet/GhostedAreaSpec.hh"

#include <vector>
#include <string>


namespace fastjet {
  class JetDefinition;
}
class FastJetFWLiteWrapper
{  
 public:
  FastJetFWLiteWrapper();
  ~FastJetFWLiteWrapper();
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);


 private:         
 

  int theMode_;

  double thePtMin_;
  double theRparam_;
  double theDcut_;
  int theNjets_;
  // Subtraction-Parameters:
  bool theDoSubtraction_;
  double theGhost_EtaMax_;
  int theActive_Area_Repeats_;
  //fastjet::ActiveAreaSpec theArea_Spec;
  double theGhostArea_;
  double theMedian_Pt_Per_Area_;  
  fastjet::JetDefinition* mJetDefinition;
  fastjet::GhostedAreaSpec* mActiveArea;
 

 public:  
  // Set methods --------------------------------------------
  void setPtMin (double aPtMin){thePtMin_=aPtMin;}
  void setRParam (double aRparam){ theRparam_=aRparam;} 

  // Get methods --------------------------------------------
  double getPtMin (){return thePtMin_ ;}
  double getRParam(){return  theRparam_;} 
};


#endif

