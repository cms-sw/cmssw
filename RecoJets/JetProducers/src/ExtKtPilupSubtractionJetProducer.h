#ifndef JetProducers_ExtKtPilupSubtractionJetProducer_h
#define JetProducers_ExtKtPilupSubtractionJetProducer_h

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

#include "BasePilupSubtractionJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/ExtKtJetAlgorithmWrapper.h"



namespace cms
{
  class ExtKtPilupSubtractionJetProducer : public cms::BasePilupSubtractionJetProducer
  {
  public:

    ExtKtPilupSubtractionJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~ExtKtPilupSubtractionJetProducer() {}
    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);

  private:
    /** Reconstruction algorithm*/
    ExtKtJetAlgorithmWrapper alg_;
  };
}

#endif
