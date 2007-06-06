#ifndef HLTMakeSummaryObjects_h
#define HLTMakeSummaryObjects_h

/** \class HLTMakeSummaryObjects
 *
 *  
 *  This class is an EDProducer making the HLT summary objects (path
 *  objects and global object).
 *
 *  $Date: 2006/08/14 15:26:42 $
 *  $Revision: 1.10 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

//
// class declaration
//

class HLTMakeSummaryObjects : public edm::EDProducer {

  public:
    explicit HLTMakeSummaryObjects(const edm::ParameterSet&);
    ~HLTMakeSummaryObjects();
    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:
    // the pointer to the current TriggerNamesService
    edm::service::TriggerNamesService* tns_;

};

#endif //HLTMakeSummaryObjects_h
