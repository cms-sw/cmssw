#ifndef JetProducers_CDFMidpointJetProducer_h
#define JetProducers_CDFMidpointJetProducer_h

/** \class CDFMidpointJetProducer
 *
 * CDFMidpointJetProducer is the EDProducer subclass which runs 
 * the CMSmidpointAlgorithm jet-finding algorithm.
 *
 * \author Marc Paterno, Fermilab
 *
 * \version   1st Version Apr. 6, 2005  
 * \version   F.Ratnikov, Mar. 8, 2006. Work from Candidate
 * $Id: CDFMidpointJetProducer.h,v 1.1 2007/06/30 17:24:06 fedor Exp $
 *
 ************************************************************/

#include "BaseJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CDFMidpointAlgorithmWrapper.h"

namespace cms
{
  class CDFMidpointJetProducer : public cms::BaseJetProducer
  {
  public:

    CDFMidpointJetProducer(const edm::ParameterSet& ps);

    virtual ~CDFMidpointJetProducer() {};


    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);
  private:
    CDFMidpointAlgorithmWrapper alg_;
  };
}


#endif
