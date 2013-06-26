/** \class TriggerSummaryProducerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2012/08/09 20:00:20 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerRAW.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include<vector>

//
// constructors and destructor
//
TriggerSummaryProducerRAW::TriggerSummaryProducerRAW(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName"))
{
  if (pn_=="@") {
    
    edm::Service<edm::service::TriggerNamesService> tns;
    if (tns.isAvailable()) {
      pn_ = tns->getProcessName();
    } else {
      edm::LogError("TriggerSummaryProducerRaw") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }
  }

  LogDebug("TriggerSummaryProducerRaw") << "Using process name: '" << pn_ <<"'";
  produces<trigger::TriggerEventWithRefs>();

  // Tell the getter what type of products to get and
  // also the process to get them from
  getterOfProducts_ = edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs>(edm::ProcessMatch(pn_), this);
  callWhenNewProductsRegistered(getterOfProducts_);
}

TriggerSummaryProducerRAW::~TriggerSummaryProducerRAW()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryProducerRAW::produce(edm::Event& iEvent, const edm::EventSetup&)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs;
   getterOfProducts_.fillHandles(iEvent, fobs);

   const unsigned int nfob(fobs.size());
   LogDebug("TriggerSummaryProducerRaw") << "Number of filter objects found: " << nfob;

   // construct single RAW product
   auto_ptr<TriggerEventWithRefs> product(new TriggerEventWithRefs(pn_,nfob));
   for (unsigned int ifob=0; ifob!=nfob; ++ifob) {
     const string& label    (fobs[ifob].provenance()->moduleLabel());
     const string& instance (fobs[ifob].provenance()->productInstanceName());
     const string& process  (fobs[ifob].provenance()->processName());
     const InputTag tag(label,instance,process);
     LogTrace("TriggerSummaryProducerRaw")
       << ifob << " " << tag << endl
       << " Sizes: "
       << " 1/" << fobs[ifob]->photonSize()
       << " 2/" << fobs[ifob]->electronSize()
       << " 3/" << fobs[ifob]->muonSize()
       << " 4/" << fobs[ifob]->jetSize()
       << " 5/" << fobs[ifob]->compositeSize()
       << " 6/" << fobs[ifob]->basemetSize()
       << " 7/" << fobs[ifob]->calometSize()

       << " 8/" << fobs[ifob]->pixtrackSize()
       << " 9/" << fobs[ifob]->l1emSize()
       << " A/" << fobs[ifob]->l1muonSize()
       << " B/" << fobs[ifob]->l1jetSize()
       << " C/" << fobs[ifob]->l1etmissSize()
       << " D/" << fobs[ifob]->l1hfringsSize()
       << " E/" << fobs[ifob]->pfjetSize()
       << " F/" << fobs[ifob]->pftauSize()
       << endl;
     product->addFilterObject(tag,*fobs[ifob]);
   }

   // place product in Event
   OrphanHandle<TriggerEventWithRefs> ref = iEvent.put(product);
   LogTrace("TriggerSummaryProducerRaw") << "Number of filter objects packed: " << ref->size();

   return;
}
