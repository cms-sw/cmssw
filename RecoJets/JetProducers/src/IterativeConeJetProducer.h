#ifndef JetProducers_IterativeConeJetProducer_h
#define JetProducers_IterativeConeJetProducer_h

/** \class IterativeConeJetProducer
 *
 * IterativeConeJetProducer is the EDProducer subclass which runs 
 * the CMSIterativeConeAlgorithm jet-finding algorithm.
 *
 ************************************************************/

#include "BaseJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CMSIterativeConeAlgorithm.h"

namespace cms
{
  class IterativeConeJetProducer : public cms::BaseJetProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    IterativeConeJetProducer(const edm::ParameterSet& ps);

    virtual ~IterativeConeJetProducer() {}

    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);

  private:
    CMSIterativeConeAlgorithm alg_;
  };
}


#endif
