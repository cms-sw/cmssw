#ifndef JetProducers_MidpointJetProducer_h
#define JetProducers_MidpointJetProducer_h

/** \class MidpointJetProducer
 *
 * MidpointJetProducer is the EDProducer subclass which runs 
 * the CMSmidpointAlgorithm jet-finding algorithm.
 *
 * \author Marc Paterno, Fermilab
 *
 * \version   1st Version Apr. 6, 2005  
 * \version   F.Ratnikov, Mar. 8, 2006. Work from Candidate
 * $Id: MidpointJetProducer.h,v 1.7 2006/08/22 22:11:40 fedor Exp $
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetProducers/interface/BaseJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CMSMidpointAlgorithm.h"

namespace cms
{
  class MidpointJetProducer : public cms::BaseJetProducer
  {
  public:

    MidpointJetProducer(const edm::ParameterSet& ps);

    virtual ~MidpointJetProducer() {};


    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, 
			       JetReco::OutputCollection* fOutput);
  private:
    CMSMidpointAlgorithm alg_;
  };
}


#endif
