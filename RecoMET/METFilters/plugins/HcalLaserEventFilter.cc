// -*- C++ -*-
//
// Package:    HcalLaserEventFilter
// Class:      HcalLaserEventFilter
// 
/**\class HcalLaserEventFilter HcalLaserEventFilter.cc UserCode/HcalLaserEventFilter/src/HcalLaserEventFilter.cc

 Description: Filter for removing Hcal laser events

 Implementation:
 Filter allows users to remove specific (run,event) pairs that have been identified as noise
It also allows users to remove events in which the number of HBHE rechits exceeds a given threshold (5000 by default).  

*/
//
// Original Author:  Jeff Temple (temple@cern.ch)
//         Created:  Thu Nov 17 12:44:22 EST 2011
// $Id: HcalLaserEventFilter.cc,v 1.1 2012/03/21 14:43:35 lhx Exp $
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

// Use for HBHERecHitCollection
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HcalLaserEventFilter : public edm::EDFilter {
   public:
      explicit HcalLaserEventFilter(const edm::ParameterSet&);
      ~HcalLaserEventFilter();

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
  
  // Filter option 1:  veto events by run, event number
  const bool vetoByRunEventNumber_;
  std::vector<std::pair<uint,uint> > RunEventData_;
  
  // Filter option 2:  veto events by HBHE occupancy
  const bool vetoByHBHEOccupancy_;
  const uint minOccupiedHBHE_;

  // Allow for debugging information to be printed
  const bool debug_;
  // Reverse the filter decision (so instead of selecting only good events, it 
  // will select only events that fail the filter conditions -- useful for studying 
  // bad events.)
  const bool reverseFilter_;

  // InputTag for HBHE rechits
  const edm::InputTag hbheInputLabel_;

  const bool taggingMode_;
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
HcalLaserEventFilter::HcalLaserEventFilter(const edm::ParameterSet& iConfig)
  
  // Get values from python cfg file
  : vetoByRunEventNumber_ (iConfig.getUntrackedParameter<bool>("vetoByRunEventNumber",true))
  , vetoByHBHEOccupancy_  (iConfig.getUntrackedParameter<bool>("vetoByHBHEOccupancy",false))
  , minOccupiedHBHE_      (iConfig.getUntrackedParameter<uint>("minOccupiedHBHE",5000))
  , debug_                (iConfig.getUntrackedParameter<bool>("debug",false))
  , reverseFilter_        (iConfig.getUntrackedParameter<bool>("reverseFilter",false)) 
  , hbheInputLabel_       (iConfig.getUntrackedParameter<edm::InputTag>("hbheInputLabel",edm::InputTag("hbhereco")))

  , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
{
  std::vector<uint> dummy; // dummy empty vector
  dummy.clear();

  std::vector<uint> temprunevt   = iConfig.getUntrackedParameter<std::vector<uint> >("BadRunEventNumbers",dummy);

  // Make (run,evt) pairs for storing bad events
  // Make this a map for better search performance?
  for (uint i=0;i+1<temprunevt.size();i+=2)
    {
      uint run=temprunevt[i];
      uint evt=temprunevt[i+1];
      RunEventData_.push_back(std::make_pair(run,evt));
    }
 
   produces<bool>();
}


HcalLaserEventFilter::~HcalLaserEventFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HcalLaserEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   bool filterDecision=true;
   
   //if (debug_) std::cout <<"<HcalLaserEventFilter> Run = "<<iEvent.id().run()<<" Event = "<<iEvent.id().event()<<std::endl;

   // Veto events by run/event numbers
   if (vetoByRunEventNumber_)
     {
       for (uint i=0;i<RunEventData_.size();++i)
	 {
	   if (iEvent.id().run()==RunEventData_[i].first &&
	       iEvent.id().event()==RunEventData_[i].second)
	     {
	       if (debug_) std::cout<<"\t<HcalLaserEventFilter> Filtering bad event;  Run "<<iEvent.id().run()<<" Event = "<<iEvent.id().event()<<std::endl;
	       filterDecision=false;
	       break;
	     }
	 }
     } // if (vetoByRunEventNumber_)

   //Veto events by HBHE rechit collection size
   if (vetoByHBHEOccupancy_)
     {
       edm::Handle<HBHERecHitCollection> hbheRecHits;
       if (iEvent.getByLabel(hbheInputLabel_,hbheRecHits))
	 {
	   //if (debug_) std::cout <<"Rechit size = "<<hbheRecHits->size()<<"  threshold = "<<minOccupiedHBHE_<<std::endl;
	   if (hbheRecHits->size()>=minOccupiedHBHE_)
	     {
	       if (debug_) std::cout <<"<HcalLaserEventFilter>  Filtering because of large HBHE rechit size; "<<hbheRecHits->size()<<" rechits is greater than or equal to the allowed maximum of "<<minOccupiedHBHE_<<std::endl;
	       filterDecision=false;
	     }
	 }
     }// if (vetoByHBHEOccupancy_)

   // Reverse decision, if specified by user
   if (reverseFilter_)
     filterDecision=!filterDecision;

   iEvent.put( std::auto_ptr<bool>(new bool(filterDecision)) );
  
   return taggingMode_ || filterDecision; 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalLaserEventFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalLaserEventFilter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
bool 
HcalLaserEventFilter::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
HcalLaserEventFilter::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
HcalLaserEventFilter::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
HcalLaserEventFilter::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HcalLaserEventFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalLaserEventFilter);
