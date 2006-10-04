#ifndef HLTProdCand_h
#define HLTProdCand_h

/** \class HLTProdCand
 *
 *  
 *  This class is a EDProducer producing some collections of
 *  reconstructed objetcs based on the Candidate model
 *
 *  $Date: 2006/08/14 16:29:11 $
 *  $Revision: 1.13 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class HLTProdCand : public edm::EDProducer {

   public:
      explicit HLTProdCand(const edm::ParameterSet&);
      ~HLTProdCand();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag jetsTag_; // MC truth jets
      edm::InputTag metsTag_; // MC truth mets

};

#endif //HLTProdCand_h
