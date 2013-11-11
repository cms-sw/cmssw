/** \class HLTSummaryFilter
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLTSummaryFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTSummaryFilter::HLTSummaryFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  summaryTag_  (iConfig.getParameter<edm::InputTag>("summary")),
  summaryToken_(consumes<trigger::TriggerEvent>(summaryTag_  )),
  memberTag_   (iConfig.getParameter<edm::InputTag>("member" )),
  cut_         (iConfig.getParameter<std::string>  ("cut"    )),
  min_N_       (iConfig.getParameter<int>          ("minN"   )),
  select_      (cut_                                          )
{
  edm::LogInfo("HLTSummaryFilter")
     << "Summary/member/cut/ncut : "
     << summaryTag_.encode() << " "
     << memberTag_.encode() << " "
     << cut_<< " " << min_N_ ;
}

HLTSummaryFilter::~HLTSummaryFilter()
{
}

//
// member functions
//
void
HLTSummaryFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  // # trigger summary
  desc.add<edm::InputTag>("summary",edm::InputTag("hltTriggerSummaryAOD","","HLT"));
  // # filter or collection
  desc.add<edm::InputTag>("member",edm::InputTag("hlt1jet30","","HLT"));
  // # cut on trigger object
  desc.add<std::string>("cut","pt>80");
  // # min. # of passing objects needed
  desc.add<int>("minN",1);
  descriptions.add("hltSummaryFilter", desc);

}

// ------------ method called to produce the data  ------------
bool
HLTSummaryFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   Handle<TriggerEvent> summary;
   iEvent.getByToken(summaryToken_,summary);

   if (!summary.isValid()) {
     LogError("HLTSummaryFilter") << "Trigger summary product "
				  << summaryTag_.encode()
				  << " not found! Filter returns false always";
     return false;
   }

   size_type n(0);
   size_type index(0);

   // check if we want to look at a filter and its passing physics objects
   index=summary->filterIndex(memberTag_);
   if (index<summary->sizeFilters()) {
     const Keys& KEYS (summary->filterKeys(index));
     const size_type n1(KEYS.size());
     for (size_type i=0; i!=n1; ++i) {
       const TriggerObject& TO( summary->getObjects().at(KEYS[i]) );
       if (select_(TO)) n++;
     }
     const bool accept(n>=min_N_);
     LogInfo("HLTSummaryFilter")
       << " Filter objects: " << n << "/" << n1;
     return accept;
   }

   // check if we want to cut on all physics objects of a full "L3" collection
   index=summary->collectionIndex(memberTag_);
   if (index<summary->sizeCollections()) {
     const Keys& KEYS (summary->collectionKeys());
     const size_type n0 (index == 0? 0 : KEYS.at(index-1));
     const size_type n1 (KEYS.at(index));
     for (size_type i=n0; i!=n1; ++i) {
       const TriggerObject& TO( summary->getObjects().at(i) );
       if (select_(TO)) n++;
     }
     const bool accept(n>=min_N_);
     LogInfo("HLTSummaryFilter")
       << " Collection objects: " << n << "/" <<n1-n0;
     return accept;
   }

   // can't help you, bailing out!
   const bool accept (false);
   LogInfo("HLTSummaryFilter") << " Default decision: " << accept;
   return accept;

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTSummaryFilter);
