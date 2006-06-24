#ifndef HLTProdCand_h
#define HLTProdCand_h

/** \class HLTProdCand
 *
 *  
 *  This class is a EDProducer producing some collections of
 *  reconstructed objetcs based on the Candidate model
 *
 *  $Date: 2006/06/17 00:18:35 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTProdCand : public edm::EDProducer {

   public:
      explicit HLTProdCand(const edm::ParameterSet&);
      ~HLTProdCand();

      virtual void produce(edm::Event&, const edm::EventSetup&);

};

#endif //HLTProdCand_h
