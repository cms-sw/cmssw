/** \class EcalCompactTrigPrimProducer
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

class EcalCompactTrigPrimProducer: public edm::EDProducer {

public:
  EcalCompactTrigPrimProducer(const edm::ParameterSet& ps);
  virtual ~EcalCompactTrigPrimProducer(){}
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

private:

  // Input collection
  edm::InputTag inCollection_;

  /*
   * output collections
   */
  std::string outCollection_;
  
};

#include "FWCore/Framework/interface/MakerMacros.h"  
DEFINE_FWK_MODULE( EcalCompactTrigPrimProducer );
