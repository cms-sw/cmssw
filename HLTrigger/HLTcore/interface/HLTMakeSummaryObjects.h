#ifndef HLTMakeSummaryObjects_h
#define HLTMakeSummaryObjects_h

/** \class HLTMakeSummaryObjects
 *
 *  
 *  This class is an EDProducer making the HLT summary objects (path
 *  objects and global object).
 *
 *  $Date: 2006/06/25 19:03:02 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTMakeSummaryObjects : public edm::EDProducer {

   public:
      explicit HLTMakeSummaryObjects(const edm::ParameterSet&);
      ~HLTMakeSummaryObjects();

   virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     unsigned int nTrig_; // number of trigger paths in schedule
     // more precisely, highest index,+1, when counting from 0!
     // attention: should be taken from TriggerResults Object!

};

#endif //HLTMakeSummaryObjects_h
