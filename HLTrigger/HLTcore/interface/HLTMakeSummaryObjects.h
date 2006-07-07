#ifndef HLTMakeSummaryObjects_h
#define HLTMakeSummaryObjects_h

/** \class HLTMakeSummaryObjects
 *
 *  
 *  This class is an EDProducer making the HLT summary objects (path
 *  objects and global object).
 *
 *  $Date: 2006/06/26 00:19:11 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
#include<vector>

//
// class decleration
//

class HLTMakeSummaryObjects : public edm::EDProducer {

   public:
      explicit HLTMakeSummaryObjects(const edm::ParameterSet&);
      ~HLTMakeSummaryObjects();

   virtual void produce(edm::Event&, const edm::EventSetup&);

   std::vector<std::string> names_; // the (path) names (used as product instance names for path objects)

};

#endif //HLTMakeSummaryObjects_h
