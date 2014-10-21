/** \class EcalCompactTrigPrimProducer
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class EcalCompactTrigPrimProducer: public edm::stream::EDProducer<> {

public:
  EcalCompactTrigPrimProducer(const edm::ParameterSet& ps);
  virtual ~EcalCompactTrigPrimProducer(){}
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

private:

  edm::EDGetTokenT<EcalTrigPrimDigiCollection>  inCollectionToken_;

  /*
   * output collections
   */
  std::string outCollection_;
  
};

#include "FWCore/Framework/interface/MakerMacros.h"  
DEFINE_FWK_MODULE( EcalCompactTrigPrimProducer );
