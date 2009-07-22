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
 * $Id: SISConeJetProducer.h,v 1.1 2007/06/30 17:24:07 fedor Exp $
 *
 ************************************************************/

#include "BaseJetProducer.h"
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
