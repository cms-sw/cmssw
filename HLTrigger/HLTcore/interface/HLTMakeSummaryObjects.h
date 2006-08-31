#ifndef HLTMakeSummaryObjects_h
#define HLTMakeSummaryObjects_h

/** \class HLTMakeSummaryObjects
 *
 *  
 *  This class is an EDProducer making the HLT summary objects (path
 *  objects and global object).
 *
 *  $Date: 2006/07/07 13:24:30 $
 *  $Revision: 1.3 $
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

  private:
    // the (path) names (used as product instance names for path objects)
    // - will be taken from triger names service (tns) in c'tor.
    std::vector<std::string> names_;

};

#endif //HLTMakeSummaryObjects_h
