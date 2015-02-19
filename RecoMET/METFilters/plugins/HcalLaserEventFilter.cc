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
//
//


// system include files
#include <memory>
#include <sstream>
#include <iostream>
#include <string>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Use for HBHERecHitCollection
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
//
// class declaration
//

class HcalLaserEventFilter : public edm::EDFilter {
   public:
      explicit HcalLaserEventFilter(const edm::ParameterSet&);
      ~HcalLaserEventFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

  std::vector<int>  GetCMSSWVersion(std::string const&);
  bool IsGreaterThanMinCMSSWVersion(std::vector<int> const&);

      // ----------member data ---------------------------

  // Filter option 1:  veto events by run, event number
  const bool vetoByRunEventNumber_;
  std::vector<std::pair<edm::RunNumber_t,edm::EventNumber_t> > RunEventData_;

  // Filter option 2:  veto events by HBHE occupancy
  const bool vetoByHBHEOccupancy_;
  const unsigned int minOccupiedHBHE_;

  // Allow for debugging information to be printed
  const bool debug_;
  // Reverse the filter decision (so instead of selecting only good events, it
  // will select only events that fail the filter conditions -- useful for studying
  // bad events.)
  const bool reverseFilter_;

  // InputTag for HBHE rechits
  const edm::InputTag hbheInputLabel_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
  const edm::InputTag hcalNoiseSummaryLabel_;
  edm::EDGetTokenT<HcalNoiseSummary> hcalNoiseSummaryToken_;

  const bool taggingMode_;

  // Control maximum number of error messages to display
  int errorcount;
  int maxerrormessage_;

  // Decide whether to use the HcalNoiseSummary to get the RecHit info, or to use the RecHit Collection itself
  bool useHcalNoiseSummary_;
  bool forceUseRecHitCollection_;
  bool forceUseHcalNoiseSummary_;
  std::vector<int> minVersion_;
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
  , minOccupiedHBHE_            (iConfig.getUntrackedParameter<unsigned int>("minOccupiedHBHE",5000))
  , debug_                      (iConfig.getUntrackedParameter<bool>("debug",false))
  , reverseFilter_              (iConfig.getUntrackedParameter<bool>("reverseFilter",false))
  , hbheInputLabel_             (iConfig.getUntrackedParameter<edm::InputTag>("hbheInputLabel",edm::InputTag("hbhereco")))
  , hbheToken_             (mayConsume<HBHERecHitCollection>(hbheInputLabel_))

  , hcalNoiseSummaryLabel_      (iConfig.getUntrackedParameter<edm::InputTag>("hcalNoiseSummaryLabel",edm::InputTag("hcalnoise")))
  , hcalNoiseSummaryToken_      (mayConsume<HcalNoiseSummary>(hcalNoiseSummaryLabel_))
  , taggingMode_                (iConfig.getParameter<bool>("taggingMode"))
  , maxerrormessage_            (iConfig.getUntrackedParameter<int>("maxerrormessage",1))
  , forceUseRecHitCollection_   (iConfig.getUntrackedParameter<bool>("forceUseRecHitCollection",false))
  , forceUseHcalNoiseSummary_   (iConfig.getUntrackedParameter<bool>("forceUseHcalNoiseSummary",false))

{
  std::vector<unsigned int> dummy; // dummy empty vector
  dummy.clear();

  std::vector<unsigned int> temprunevt   = iConfig.getUntrackedParameter<std::vector<unsigned int> >("BadRunEventNumbers",dummy);

  // Make (run,evt) pairs for storing bad events
  // Make this a map for better search performance?
  for (unsigned int i=0;i+1<temprunevt.size();i+=2)
    {
      edm::RunNumber_t run=temprunevt[i];
      edm::EventNumber_t evt=temprunevt[i+1];
      RunEventData_.push_back(std::make_pair(run,evt));
    }
  errorcount=0;
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

   if (debug_) std::cout <<"<HcalLaserEventFilter> Run = "<<iEvent.id().run()<<" Event = "<<iEvent.id().event()<<std::endl;

   // Veto events by run/event numbers
   if (vetoByRunEventNumber_)
     {
       for (unsigned int i=0;i<RunEventData_.size();++i)
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
       // The decision on whether or not to use the noise summary is made based on the CMSSW version.
       // As of CMSSW_5_2_0, the HcalNoiseSummary contains the total number of HBHE hits, as well as number of hits > 1.5 GeV and some other info.
       // The boolean 'forceUseRecHitCollection_' can be used to override this automatic behavior, and to use the RecHit collection itself, regardless of CMSSW version.


       //////////////////////////////////////////////////////////
       //
       //  Apply Filtering based on RecHit information in HBHERecHitcollection
       //
       ////////////////////////////////////////////////////////////


       if (useHcalNoiseSummary_==false || forceUseRecHitCollection_==true)
	 {
	   edm::Handle<HBHERecHitCollection> hbheRecHits;
	   if (iEvent.getByToken(hbheToken_,hbheRecHits))
	     {
	       if (debug_) std::cout <<"Rechit size = "<<hbheRecHits->size()<<"  threshold = "<<minOccupiedHBHE_<<std::endl;
	       if (hbheRecHits->size()>=minOccupiedHBHE_)
		 {
		   if (debug_) std::cout <<"<HcalLaserEventFilter>  Filtering because of large HBHE rechit size; "<<hbheRecHits->size()<<" rechits is greater than or equal to the allowed maximum of "<<minOccupiedHBHE_<<std::endl;
		   filterDecision=false;
		 }
	     }
	   else
	     {
	       if (debug_ && errorcount<maxerrormessage_)
		 std::cout <<"<HcalLaserEventFilter::Error> No valid HBHERecHitCollection with label '"<<hbheInputLabel_<<"' found"<<std::endl;
	       ++errorcount;
	     }
	 }

       //////////////////////////////////////////////////////////
       //
       //  Apply Filtering based on RecHit information in HcalNoiseSummary object
       //
       ////////////////////////////////////////////////////////////
       else if (useHcalNoiseSummary_==true || forceUseHcalNoiseSummary_==true)
	 {
	   Handle<HcalNoiseSummary> hSummary;
	   if (iEvent.getByToken(hcalNoiseSummaryToken_,hSummary)) // get by label, usually with label 'hcalnoise'
	     {
	       if (debug_)  std::cout << " RECHIT SIZE (from HcalNoiseSummary) = "<<hSummary->GetRecHitCount()<<"  threshold = "<<minOccupiedHBHE_<<std::endl;
	       if (hSummary->GetRecHitCount() >= (int)minOccupiedHBHE_)
		 {
		   if (debug_) std::cout <<"<HcalLaserEventFilter>  Filtering because of large HBHE rechit size in HcalNoiseSummary; "<<hSummary->GetRecHitCount()<<" rechits is greater than or equal to the allowed maximum of "<<minOccupiedHBHE_<<std::endl;
		   filterDecision=false;
		 }
	     }
	   else
	     {
	       if (debug_ && errorcount<maxerrormessage_)
		 std::cout <<"<HcalLaserEventFilter::Error> No valid HcalNoiseSummary with label '"<<hcalNoiseSummaryLabel_<<"' found"<<std::endl;
	       ++errorcount;
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
  // Specify the minimum release that has the rechit counts in the HcalNoiseSummary object.
  // If current release >= that release, then HcalNoiseSummary will be used.  Otherwise, Rechit collection will be used.
  std::string minRelease="\"CMSSW_5_2_0\"";

  minVersion_=GetCMSSWVersion(minRelease);
  std::vector <int> currentVersion=GetCMSSWVersion(edm::getReleaseVersion());

  if (IsGreaterThanMinCMSSWVersion(currentVersion)) // current Version is >= minVersion_
    useHcalNoiseSummary_=true;
  else
    useHcalNoiseSummary_=false;
}

// ------------ method called once each job just after ending the event loop  ------------
void
HcalLaserEventFilter::endJob() {
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

std::vector<int>  HcalLaserEventFilter::GetCMSSWVersion(std::string const& instring)
{
  std::vector <int> temp;


  std::string prefix;
  std::string v1, v2, v3;

  std::istringstream oss(instring);
  getline(oss, prefix,'_');
  getline(oss, v1,'_');
  getline(oss, v2,'_');
  getline(oss, v3,'_');

  std::stringstream buffer(v1);
  int t;
  buffer>>t;
  temp.push_back(t);
  buffer.str();
  buffer<<v2;
  buffer>>t;
  temp.push_back(t);
  buffer.str();
  buffer<<v3;
  buffer>>t;
  temp.push_back(t);

  //std::cout <<"PREFIX = "<<prefix<<" "<<temp[0]<<" "<<temp[1]<<" "<<temp[2]<<std::endl;
  //( ex:  PREFIX = "CMSSW 5 5 5  )
  return temp;
}

bool HcalLaserEventFilter::IsGreaterThanMinCMSSWVersion(std::vector<int> const& currentVersion)
{
  // Returns false if current version is less than min version
  // Otherwise, returns true
  // Assumes CMSSW versioning X_Y_Z



  // Compare X
  if (currentVersion[0]<minVersion_[0]) return false;
  if (currentVersion[0]>minVersion_[0]) return true;
  // If neither is true, first value of CMSSW versions are the same

  // Compare Y
  if (currentVersion[1]<minVersion_[1]) return false;
  if (currentVersion[1]>minVersion_[1]) return true;

  // Compare Z
  if (currentVersion[2]<minVersion_[2]) return false;
  if (currentVersion[2]>minVersion_[2]) return true;

  return true; // versions are identical
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalLaserEventFilter);
