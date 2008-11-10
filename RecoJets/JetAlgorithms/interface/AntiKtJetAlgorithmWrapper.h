#ifndef JetAlgorithms_AntiKtJetAlgorithmWrapper_h
#define JetAlgorithms_AntiKtJetAlgorithmWrapper_h

/** \class AntiKtJetAlgorithmWrapper
 *
 * AntiKtJetAlgorithmWrapper is the Wrapper subclass which runs
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
 * of CMSSW.
 * Adapted to anti-kt from the kt version by Salvatore Rappoccio. 
 *
 * \version   1st Version Nov. 6 2006
 * \version   2nd redesigned by F.Ratnikov (UMd) Aug. 1, 2007
 * 
 *
 *  
 *
 ************************************************************/

#include "RecoJets/JetAlgorithms/interface/FastJetBaseWrapper.h"

class AntiKtJetAlgorithmWrapper : public FastJetBaseWrapper {
 public:
  AntiKtJetAlgorithmWrapper(const edm::ParameterSet& fConfig);
  virtual ~AntiKtJetAlgorithmWrapper();
 protected:
};

#endif
