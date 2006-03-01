#ifndef JetProducers_MidpointJetProducer2_h
#define JetProducers_MidpointJetProducer2_h

/** \class MidpointJetProducer2
 *
 * MidpointJetProducer2 is the EDProducer subclass which runs 
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

#include "RecoJets/JetAlgorithms/interface/CMSMidpointAlgorithm.h"

namespace cms
{
  class MidpointJetProducer2 : public edm::EDProducer
  {
  public:

    // The following is not yet used, but will be the primary
    // constructor when the parameter set system is available.
    //
    explicit MidpointJetProducer2(const edm::ParameterSet& ps);

    virtual ~MidpointJetProducer2();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CMSMidpointAlgorithm alg_;
    std::string src_;
  };
}


#endif
