// -*- C++ -*-
//
// Package:    TriggerTypeFilter
// Class:      TriggerTypeFilter
// 
/**\class TriggerTypeFilter TriggerTypeFilter.cc filter/TriggerTypeFilter/src/TriggerTypeFilter.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

//
// class declaration
//

class TriggerTypeFilter : public HLTFilter {
public:
  explicit TriggerTypeFilter(const edm::ParameterSet&);
  ~TriggerTypeFilter();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  std::string     DataLabel_ ;
  unsigned short  TriggerFedId_;
  unsigned short  SelectedTriggerType_;
  
};

//
// constants, enums and typedefs
//
enum GapFilterConstants{

  EMPTY_EVENTSIZE        = 32,

  H_FEDID_B              = 8,
  H_FEDID_MASK           = 0xFFF,
   
  H_TTYPE_B              = 56,
  H_TTYPE_MASK           = 0xF    

};

//
// constructors and destructor
//
TriggerTypeFilter::TriggerTypeFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  DataLabel_     = iConfig.getParameter<std::string>("InputLabel");
  TriggerFedId_  = static_cast<unsigned short>(iConfig.getParameter<int>("TriggerFedId"));
  SelectedTriggerType_   = static_cast<unsigned short>(iConfig.getParameter<int>("SelectedTriggerType"));

}


TriggerTypeFilter::~TriggerTypeFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TriggerTypeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<FEDRawDataCollection> rawdata;  
  iEvent.getByLabel(DataLabel_,rawdata);

  // get fed raw data and SM id
  const FEDRawData & fedData = rawdata->FEDData(TriggerFedId_);

  uint64_t * pData = (uint64_t *)(fedData.data());

  // First Header Word of fed block
  unsigned short triggerType       = ((*pData)>>H_TTYPE_B)   & H_TTYPE_MASK;

  return (triggerType == SelectedTriggerType_) ? true : false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
TriggerTypeFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TriggerTypeFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerTypeFilter);
