#ifndef KtJetProducer_h
#define KtJetProducer_h

/** \class KtJetProducer
 *
 * KtJetProducer is the EDProducer subclass which runs 
 * the CMSKtJetAlgorithm jet-finding algorithm.
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version Apr. 22, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoJets/JetAlgorithms/interface/CMSKtJetAlgorithm.h"

namespace cms
{
  class KtJetProducer : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit KtJetProducer(const edm::ParameterSet& ps);

    /**Default destructor*/
    virtual ~KtJetProducer();
    /**Produces the EDM products, .e Kt Jets*/
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    /** Reconstruction algorithm*/
    CMSKtJetAlgorithm alg_;
  };
}

#endif
