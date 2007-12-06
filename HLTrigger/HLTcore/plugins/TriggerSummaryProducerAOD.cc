/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2007/08/07 18:42:19 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName")),
  selector_(edm::ProcessNameSelector(pn_)),
  tns_(),
  collections_(ps.getParameter<std::vector<edm::InputTag> >("collections")),
  filters_(ps.getParameter<std::vector<edm::InputTag> >("filters")),
  offset_(),
  fobs_()
{

  if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
    // get tns pointer
    tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
    if (tns_!=0) {
      pn_=tns_->getProcessName();
    } else {
      LogDebug("") << "HLT Error: TriggerNamesService pointer = 0!";
    }
  } else {
    LogDebug("") << "HLT Error: TriggerNamesService not available!";
  }

  selector_=edm::ProcessNameSelector(pn_);
  LogDebug("") << "Using process name: " << pn_;
  produces<trigger::TriggerEvent>(pn_);

}

TriggerSummaryProducerAOD::~TriggerSummaryProducerAOD()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryProducerAOD::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   /// First the L3 collections:

   // reset from previous event
   offset_.clear();

   /// Now the filter objects:

   // reset from previous event
   fobs_.clear();

   // get all filter objects created in requested process
   iEvent.getMany(selector_,fobs_);
   const size_type nfob(fobs_.size());
   LogDebug("") << "Number of filter objects found: " << nfob;

   // construct single AOD product
   auto_ptr<TriggerEvent> product(new TriggerEvent(nfob));
   //   for (size_type ifob=0; ifob!=nfob; ++ifob) {
   //     product->addFilterObject(fobs_[ifob].provenance()->moduleLabel(),*fobs_[ifob]);
   //   }

   // place product in Event
   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   //   LogTrace("") << "Number of filter objects packed: " << ref->size();

   return;
}
