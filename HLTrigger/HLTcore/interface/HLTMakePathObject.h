#ifndef HLTMakePathObject_h
#define HLTMakePathObject_h

/** \class HLTMakePathObject
 *
 *  
 *  This class is an EDProducer making the HLT path object.
 *
 *  $Date: 2006/05/12 18:13:11 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<vector>

//
// class decleration
//

class HLTMakePathObject : public edm::EDProducer {

   public:
      explicit HLTMakePathObject(const edm::ParameterSet&);
      ~HLTMakePathObject();

   virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     std::vector<edm::InputTag> inputTags_; // tags of products
};

#endif //HLTMakePathObject_h
