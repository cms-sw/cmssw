/** \class TriggerSummaryProducerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/05/02 12:13:28 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerRAW.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

//
// constructors and destructor
//
TriggerSummaryProducerRAW::TriggerSummaryProducerRAW(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName")),
  selector_(edm::ProcessNameSelector(pn_)),
  tns_(), fobs_()
{
  if (pn_=="@") {
    // use tns
    if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
      // get tns pointer
      tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
      if (tns_!=0) {
	pn_=tns_->getProcessName();
      } else {
	LogDebug("TriggerSummaryProducerRaw") << "HLT Error: TriggerNamesService pointer = 0!";
	pn_="*";
      }
    } else {
      LogDebug("TriggerSummaryProducerRaw") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }
    selector_=edm::ProcessNameSelector(pn_);
  }

  LogDebug("TriggerSummaryProducerRaw") << "Using process name: '" << pn_ <<"'";
  produces<trigger::TriggerEventWithRefs>();

}

TriggerSummaryProducerRAW::~TriggerSummaryProducerRAW()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryProducerRAW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // reset from previous event
   fobs_.clear();

   // get all filter objects created in requested process
   iEvent.getMany(selector_,fobs_);
   const size_type nfob(fobs_.size());
   LogDebug("TriggerSummaryProducerRaw") << "Number of filter objects found: " << nfob;

   // construct single RAW product
   auto_ptr<TriggerEventWithRefs> product(new TriggerEventWithRefs(pn_,nfob));
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     const string& label    (fobs_[ifob].provenance()->moduleLabel());
     const string& instance (fobs_[ifob].provenance()->productInstanceName());
     const string& process  (fobs_[ifob].provenance()->processName());
     const InputTag tag(InputTag(label,instance,process));
     LogTrace("TriggerSummaryProducerRaw") << ifob << " " << tag;
     product->addFilterObject(tag,*fobs_[ifob]);
   }

   // place product in Event
   OrphanHandle<TriggerEventWithRefs> ref = iEvent.put(product);
   LogTrace("TriggerSummaryProducerRaw") << "Number of filter objects packed: " << ref->size();

   return;
}
