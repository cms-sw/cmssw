#ifndef JetProducers_CambridgePilupSubtractionJetProducer_h
#define JetProducers_CambridgePilupSubtractionJetProducer_h

/** \class FastJetProducer
 *
 * FastJetProducer is the EDProducer subclass which runs
 * the FastJetAlgorithm for jetfinding.
 * 
 * The FastJet package, written by Matteo Cacciari and Gavin Salam, 
 * provides a fast implementation of the longitudinally invariant kt 
 * and longitudinally invariant inclusive Cambridge/Aachen jet finders.
 * More information can be found at:
 * http://parthe.lpthe.jussieu.fr/~salam/fastjet/
 *
 * The algorithms that underlie FastJet have required considerable
 * development and are described in hep-ph/0512210. If you use
 * FastJet as part of work towards a scientific publication, please
 * include a citation to the FastJet paper.
 *
 * \author Andreas Oehler, University Karlsruhe (TH)
 * has written the FastJetProducer class
 * which uses the above mentioned package within the Framework
 * of CMSSW
 *
 * \version   1st Version Nov. 6 2006
 * 
 *
 ************************************************************/

#include "BasePilupSubtractionJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CambridgeAlgorithmWrapper.h"

namespace cms
{
  class CambridgePilupSubtractionJetProducer : public cms::BasePilupSubtractionJetProducer
  {
  public:

    CambridgePilupSubtractionJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~CambridgePilupSubtractionJetProducer() {}
    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);

  private:
    /** Reconstruction algorithm*/
    CambridgeAlgorithmWrapper alg_;
  };
}

#endif
