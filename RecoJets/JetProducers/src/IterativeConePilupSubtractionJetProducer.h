#ifndef JetProducers_IterativeConePilupSubtractionJetProducer_h
#define JetProducers_IterativeConePilupSubtractionJetProducer_h

/** \class IterativeConeJetProducer
 *
 * IterativeConeJetProducer is the EDProducer subclass which runs 
 * the CMSIterativeConeAlgorithm jet-finding algorithm.
 *
 ************************************************************/

#include "BasePilupSubtractionJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CMSIterativeConeAlgorithm.h"

namespace cms
{
  class IterativeConePilupSubtractionJetProducer : public cms::BasePilupSubtractionJetProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    IterativeConePilupSubtractionJetProducer(const edm::ParameterSet& ps);

    virtual ~IterativeConePilupSubtractionJetProducer() {}

    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);

  private:
    CMSIterativeConeAlgorithm alg_;
  };
}


#endif
