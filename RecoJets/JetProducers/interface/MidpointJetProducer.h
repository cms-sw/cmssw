#ifndef MidpointJetProducer_h
#define MidpointJetProducer_h

/** \class MidpointJetProducer
 *
 * MidpointJetProducer is the EDProducer subclass which runs 
 * the CMSmidpointAlgorithm jet-finding algorithm.
 *
 * \author Marc Paterno, Fermilab
 *
 * \version   1st Version Apr. 6, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetAlgorithms/interface/CMSmidpointAlgorithm.h"

namespace cms
{
  class MidpointJetProducer : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit MidpointJetProducer(const edm::ParameterSet& ps);

    virtual ~MidpointJetProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CMSmidpointAlgorithm alg_;
  };
}


#endif
