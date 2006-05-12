#ifndef HLTMakePathObject_h
#define HLTMakePathObject_h

/** \class HLTMakePathObject
 *
 *  
 *  This class is an EDProducer making the HLT path object.
 *
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<vector>
#include<string>

//
// class decleration
//

class HLTMakePathObject : public edm::EDProducer {

   public:
      explicit HLTMakePathObject(const edm::ParameterSet&);
      ~HLTMakePathObject();

   virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     std::vector<std::string > labels_;  // module labels
     std::vector<unsigned int> indices_; // module indices
};

#endif //HLTMakePathObject_h
