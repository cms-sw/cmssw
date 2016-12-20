// -*- C++ -*-
//
// Package:    L1TValidationEventFilter
// Class:      L1TValidationEventFilter
// 
/**\class L1TValidationEventFilter L1TValidationEventFilter.cc EventFilter/L1TRawToDigi/src/L1TValidationEventFilter.cc

Description: <one line class summary>
Implementation:
<Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <iostream>


#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"


//
// class declaration
//

class L1TValidationEventFilter : public edm::EDFilter {
public:
  explicit L1TValidationEventFilter(const edm::ParameterSet&);
  virtual ~L1TValidationEventFilter();
  
private:
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  
  // ----------member data ---------------------------
  edm::InputTag src_;
  int period_;       // validation event period

};


//
// constructors and destructor
//
L1TValidationEventFilter::L1TValidationEventFilter(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  period_( iConfig.getUntrackedParameter<int>("period", 107) )
{
  //now do what ever initialization is needed

}


L1TValidationEventFilter::~L1TValidationEventFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
L1TValidationEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle<int> triggerCount;
  iEvent.getByLabel(InputTag(src_.label(),"triggerCount"), triggerCount);
  if (!triggerCount.isValid()) {
    LogError("L1T") << "TCDS data not unpacked: triggerCount not availble in Event.";
    return false;
  }

  bool fatEvent = (*triggerCount % period_ == 0 );

  return fatEvent;

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TValidationEventFilter::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TValidationEventFilter::endJob() {

}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TValidationEventFilter);
