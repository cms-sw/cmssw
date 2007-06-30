#ifndef JetProducers_SISConeJetProducer_h
#define JetProducers_SISConeJetProducer_h

/** \class SISConeJetProducer
 *
 * SISConeJetProducer is the EDProducer subclass which runs 
 * the SISCone jet-finding algorithm from fastjet package.
 *
 * \author Fedor Ratnikov, Maryland, June 30, 2007
 *
 * \version   1st Version Apr. 6, 2005  
 * \version   F.Ratnikov, Mar. 8, 2006. Work from Candidate
 * $Id: SISConeJetProducer.h,v 1.9 2007/03/07 18:43:43 fedor Exp $
 *
 ************************************************************/

#include "RecoJets/JetProducers/interface/BaseJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/SISConeAlgorithmWrapper.h"

namespace cms
{
  class SISConeJetProducer : public cms::BaseJetProducer
  {
  public:

    SISConeJetProducer(const edm::ParameterSet& ps);

    virtual ~SISConeJetProducer() {};


    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);
  private:
    SISConeAlgorithmWrapper alg_;
  };
}


#endif
