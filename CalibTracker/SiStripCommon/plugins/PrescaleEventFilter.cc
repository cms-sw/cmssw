// -*- C++ -*-
//
// Package:    CalibTracker/SiStripCommon
// Class:      PrescaleEventFilter
// 
/**\class PrescaleEventFilter PrescaleEventFilter.cc CalibTracker/SiStripCommon/plugins/PrescaleEventFilter.cc

 Description: Simple class to prescale events entering the Strip Tracker Calibration Tree

 Implementation:
     Largely copied from HLTrigger/HLTcore/plugins/HLTPrescaler.cc, without the need to specify a specific trigger path
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 29 Nov 2017 15:27:07 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace prescale {
  struct Efficiency {
    Efficiency(): eventCount_(0),acceptCount_(0) { }
    mutable std::atomic<unsigned int> eventCount_;
    mutable std::atomic<unsigned int> acceptCount_;
  };
}

//
// class declaration
//

class PrescaleEventFilter : public edm::stream::EDFilter<edm::GlobalCache<prescale::Efficiency> > {
   public:
      explicit PrescaleEventFilter(edm::ParameterSet const& iConfig, const prescale::Efficiency* efficiency);
      ~PrescaleEventFilter() override;

      static std::unique_ptr<prescale::Efficiency> initializeGlobalCache(edm::ParameterSet const&) {
 return std::unique_ptr<prescale::Efficiency>(new prescale::Efficiency());
      };

  
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
      static void globalEndJob(const prescale::Efficiency* efficiency);

   private:
      void beginStream(edm::StreamID) override;
      bool filter(edm::Event&, const edm::EventSetup&) override;
      void endStream() override;

      // ----------member data ---------------------------

      /// accept one in prescaleFactor_; 0 means never to accept an event
      unsigned int prescaleFactor_;

      /// event counter
      unsigned int eventCount_;

      /// accept counter
      unsigned int acceptCount_;

      /// initial offset
      unsigned int offsetCount_;
      unsigned int offsetPhase_;
  
      /// check for (re)initialization of the prescale
      bool newLumi_;

      /// "seed" used to initialize the prescale counter
      static const unsigned int prescaleSeed_ = 65537;

};

//
// constructors and destructor
//
PrescaleEventFilter::PrescaleEventFilter(const edm::ParameterSet& iConfig,const prescale::Efficiency* efficiency):
  prescaleFactor_(iConfig.getParameter<unsigned int>("prescale")),
  eventCount_(0),
  acceptCount_(0),
  offsetCount_(0),
  offsetPhase_(iConfig.getParameter<unsigned int>("offset"))
{
   //now do what ever initialization is needed

}


PrescaleEventFilter::~PrescaleEventFilter()
{
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
PrescaleEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   bool needsInit (eventCount_==0);

   if (needsInit && (prescaleFactor_ != 0)) {
     // initialize the prescale counter to the first event number multiplied by a big "seed"
      offsetCount_ = ((uint64_t) (iEvent.id().event() + offsetPhase_) * prescaleSeed_) % prescaleFactor_;
   }

   const bool result ( (prescaleFactor_ == 0) ? 
          false : ((eventCount_ + offsetCount_) % prescaleFactor_ == 0) );

   ++eventCount_;
   if (result) ++acceptCount_;
   return result;
   
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
PrescaleEventFilter::beginStream(edm::StreamID)
{
}

//_____________________________________________________________________________
void 
PrescaleEventFilter::endStream()
{
  //since these are std::atomic, it is safe to increment them
  // even if multiple endStreams are being called.
  globalCache()->eventCount_ += eventCount_;
  globalCache()->acceptCount_ += acceptCount_; 
  return;
}

//_____________________________________________________________________________
void 
PrescaleEventFilter::globalEndJob(const prescale::Efficiency* efficiency)
{
  unsigned int accept(efficiency->acceptCount_);
  unsigned int event (efficiency->eventCount_);
  edm::LogInfo("PrescaleSummary")
    << accept << "/" << event
    << " ("
    << 100.*accept/static_cast<double>(std::max(1u,event))
    << "% of events accepted).";
  return;
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PrescaleEventFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("prescale",1);
  desc.add<unsigned int>("offset",0);
  descriptions.add("prescaleEvent", desc);

}
//define this as a plug-in
DEFINE_FWK_MODULE(PrescaleEventFilter); 
