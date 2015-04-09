// -*- C++ -*-
//
// Package:    CommonTools/RecoAlgos
// Class:      BooleanFlagFilter
// 
/**\class BooleanFlagFilter BooleanFlagFilter.cc CommonTools/RecoAlgos/plugins/BooleanFlagFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Fri, 20 Mar 2015 08:05:20 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class BooleanFlagFilter : public edm::EDFilter {
   public:
      explicit BooleanFlagFilter(const edm::ParameterSet&);
      ~BooleanFlagFilter();

   private:
      virtual void beginJob() override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<bool> inputToken_;
      bool reverse_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BooleanFlagFilter::BooleanFlagFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   inputToken_ = consumes<bool>(iConfig.getParameter<edm::InputTag>("inputLabel"));
   reverse_ = iConfig.getParameter<bool>("reverseDecision");
}


BooleanFlagFilter::~BooleanFlagFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BooleanFlagFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<bool> pIn;
   iEvent.getByToken(inputToken_, pIn);
   if (!pIn.isValid())
   {
      throw edm::Exception(edm::errors::ProductNotFound) << " could not find requested flag\n";
      return true;
   }

   bool result = *pIn;
   if (reverse_)
      result = !result;

   return result;
}

// ------------ method called once each job just before starting event loop  ------------
void 
BooleanFlagFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BooleanFlagFilter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
BooleanFlagFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
BooleanFlagFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
BooleanFlagFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
BooleanFlagFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
//define this as a plug-in
DEFINE_FWK_MODULE(BooleanFlagFilter);
