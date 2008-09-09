//
// Original Author:  Marco Zanetti
//         Created:  Tue Sep  9 15:56:24 CEST 2008


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>


class BxNumberFilter : public edm::EDFilter {
public:
  explicit BxNumberFilter(const edm::ParameterSet&);
  ~BxNumberFilter();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::InputTag inputLabel;
  unsigned int goldenBXId;

};

BxNumberFilter::BxNumberFilter(const edm::ParameterSet& iConfig) {

  inputLabel = iConfig.getUntrackedParameter<edm::InputTag>("inputLabel",edm::InputTag("source"));
  goldenBXId = iConfig.getParameter<unsigned int>("BXId");
  
}


BxNumberFilter::~BxNumberFilter() { }


bool BxNumberFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

   bool result = false;

   unsigned int GTEVMId= 812;

   Handle<FEDRawDataCollection> rawdata;
   iEvent.getByLabel(inputLabel, rawdata);  
   const FEDRawData& data = rawdata->FEDData(GTEVMId);

   // Select the BX
   if ( evf::evtn::getfdlbx(data.data()) == goldenBXId ) result = true;

   
   return result;
}

// ------------ method called once each job just before starting event loop  ------------
void  BxNumberFilter::beginJob(const edm::EventSetup&) {
}

// ------------ method called once each job just after ending the event loop  ------------
void  BxNumberFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(BxNumberFilter);
