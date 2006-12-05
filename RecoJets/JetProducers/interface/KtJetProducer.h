#ifndef JetProducers_KtJetProducer_h
#define JetProducers_KtJetProducer_h

/** \class KtJetProducer
 *
 * KtJetProducer is the EDProducer subclass which runs 
 * the CMSKtJetAlgorithm jet-finding algorithm.
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 * rewritten by F.Ratnikov (UMd) Mar. 6th, 2006
 *
 * \version   1st Version Apr. 22, 2005  
 * \version   F.Ratnikov, Mar. 8, 2006. work from Candidate
 * $Id: KtJetProducer.h,v 1.8 2006/08/22 22:11:40 fedor Exp $
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoJets/JetProducers/interface/BaseJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/CMSKtJetAlgorithm.h"

namespace cms
{
  class KtJetProducer : public cms::BaseJetProducer
  {
  public:

    KtJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~KtJetProducer() {}
    /** run algorithm itself */
    virtual bool runAlgorithm (const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);

  private:
    /** Reconstruction algorithm*/
    CMSKtJetAlgorithm alg_;
  };
}

#endif
