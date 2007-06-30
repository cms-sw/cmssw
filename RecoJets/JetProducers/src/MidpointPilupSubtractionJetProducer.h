#ifndef JetProducers_MidpointPilupSubtractionJetProducer_h
#define JetProducers_MidpointPilupSubtractionJetProducer_h

/** \class MidpointJetProducer
 *
 * MidpointJetProducer is the EDProducer subclass which runs 
 * the CMSmidpointAlgorithm jet-finding algorithm.
 *
 * \author Marc Paterno, Fermilab
 *
 * \version   1st Version Apr. 6, 2005  
 * \version   F.Ratnikov, Mar. 8, 2006. Work from Candidate
 * $Id: MidpointPilupSubtractionJetProducer.h,v 1.1 2007/05/04 09:53:17 kodolova Exp $
 *
 ************************************************************/

#include "BasePilupSubtractionJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CMSMidpointAlgorithm.h"

namespace cms
{
  class MidpointPilupSubtractionJetProducer : public cms::BasePilupSubtractionJetProducer
  {
  public:

    MidpointPilupSubtractionJetProducer(const edm::ParameterSet& ps);

    virtual ~MidpointPilupSubtractionJetProducer() {};


    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);
  private:
    CMSMidpointAlgorithm alg_;
  };
}


#endif
