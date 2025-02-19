// -*- C++ -*-
//
// Package:    HighLumiFilter
// Class:      HighLumiFilter
// 
/**\class HighLumiFilter HighLumiFilter.cc DPGAnalysis/Skims/src/HighLumiFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Giuseppe Cerati,28 S-012,+41227678302,
//         Created:  Tue Jan 15 12:46:40 CET 2013
// $Id: HighLumiFilter.cc,v 1.1 2013/01/15 15:54:55 cerati Exp $
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

#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageService/interface/MessageLogger.h"

//
// class declaration
//

class HighLumiFilter : public edm::EDFilter {
   public:
      explicit HighLumiFilter(const edm::ParameterSet&);
      ~HighLumiFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual bool beginRun(edm::Run&, edm::EventSetup const&);
      virtual bool endRun(edm::Run&, edm::EventSetup const&);
      virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
      edm::InputTag lumiTag_;
      double minLumi_;
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
HighLumiFilter::HighLumiFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   lumiTag_ = iConfig.getParameter<edm::InputTag>("lumiTag");
   minLumi_ = iConfig.getParameter<double>("minLumi");

}


HighLumiFilter::~HighLumiFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HighLumiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Handle<LumiDetails> d;
   iEvent.getLuminosityBlock().getByLabel(lumiTag_,d);
   if (d.isValid()==0) return false;
   try {
     float myrawbxlumi=d->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing());
     if (myrawbxlumi<minLumi_) return false;
   } catch (Exception e) {
     if (e.categoryCode()==errors::LogicError) {
       LogWarning("HighLumiFilter") << "Caught exception from LumiDetails and rejecting event. e.what():" << e.what();
       return false;
     } else throw e;
   }
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HighLumiFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HighLumiFilter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
bool 
HighLumiFilter::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
HighLumiFilter::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
HighLumiFilter::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
HighLumiFilter::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HighLumiFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HighLumiFilter);
