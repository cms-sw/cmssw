#ifndef HLTProdCand_h
#define HLTProdCand_h

/** \class HLTProdCand
 *
 *  
 *  This class is a EDProducer producing some collections of
 *  reconstructed objetcs based on the Candidate model
 *
 *  $Date: 2006/05/18 16:57:45 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
//
// class decleration
//

class HLTProdCand : public edm::EDProducer {

   public:
      explicit HLTProdCand(const edm::ParameterSet&);
      ~HLTProdCand();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
      unsigned int n_;  // how many objects in the collections
      double  factor_;  // scale factor for faked 4-momentum
};

#endif //HLTProdCand_h
