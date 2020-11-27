/** \class TriggerSummaryProducerRAW
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerRAW.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>

//
// constructors and destructor
//
TriggerSummaryProducerRAW::TriggerSummaryProducerRAW(const edm::ParameterSet& ps)
    : pn_(ps.getParameter<std::string>("processName")), putToken_{produces<trigger::TriggerEventWithRefs>()} {
  if (pn_ == "@") {
    edm::Service<edm::service::TriggerNamesService> tns;
    if (tns.isAvailable()) {
      pn_ = tns->getProcessName();
    } else {
      edm::LogError("TriggerSummaryProducerRaw") << "HLT Error: TriggerNamesService not available!";
      pn_ = "*";
    }
  }

  LogDebug("TriggerSummaryProducerRaw") << "Using process name: '" << pn_ << "'";

  // Tell the getter what type of products to get and
  // also the process to get them from
  getterOfProducts_ = edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs>(edm::ProcessMatch(pn_), this);
  callWhenNewProductsRegistered(getterOfProducts_);
}

TriggerSummaryProducerRAW::~TriggerSummaryProducerRAW() = default;

//
// member functions
//

void TriggerSummaryProducerRAW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "@");
  descriptions.add("triggerSummaryProducerRAW", desc);
}

// ------------ method called to produce the data  ------------
void TriggerSummaryProducerRAW::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs;
  getterOfProducts_.fillHandles(iEvent, fobs);

  const unsigned int nfob(fobs.size());
  LogDebug("TriggerSummaryProducerRaw") << "Number of filter objects found: " << nfob;

  // construct single RAW product
  TriggerEventWithRefs product(pn_, nfob);
  for (unsigned int ifob = 0; ifob != nfob; ++ifob) {
    const string& label(fobs[ifob].provenance()->moduleLabel());
    const string& instance(fobs[ifob].provenance()->productInstanceName());
    const string& process(fobs[ifob].provenance()->processName());
    const InputTag tag(label, instance, process);
    LogTrace("TriggerSummaryProducerRaw")
        << ifob << " " << tag << endl
        << " Sizes: "
        << " 1/" << fobs[ifob]->photonSize() << " 2/" << fobs[ifob]->electronSize() << " 3/" << fobs[ifob]->muonSize()
        << " 4/" << fobs[ifob]->jetSize() << " 5/" << fobs[ifob]->compositeSize() << " 6/" << fobs[ifob]->basemetSize()
        << " 7/" << fobs[ifob]->calometSize()

        << " 8/" << fobs[ifob]->pixtrackSize() << " 9/" << fobs[ifob]->l1emSize() << " A/" << fobs[ifob]->l1muonSize()
        << " B/" << fobs[ifob]->l1jetSize() << " C/" << fobs[ifob]->l1etmissSize() << " D/"
        << fobs[ifob]->l1hfringsSize() << " E/" << fobs[ifob]->pfjetSize() << " F/" << fobs[ifob]->pftauSize() << " G/"
        << fobs[ifob]->pfmetSize() << " I/" << fobs[ifob]->l1tmuonSize() << " J/" << fobs[ifob]->l1tegammaSize()
        << " K/" << fobs[ifob]->l1tjetSize() << " L/" << fobs[ifob]->l1ttauSize() << " M/" << fobs[ifob]->l1tetsumSize()
        << " N/" << fobs[ifob]->l1ttkmuonSize() << " O/" << fobs[ifob]->l1ttkeleSize() << " P/"
        << fobs[ifob]->l1ttkemSize() << " Q/" << fobs[ifob]->l1tpfjetSize() << " R/" << fobs[ifob]->l1tpftauSize()
        << " S/" << fobs[ifob]->l1thpspftauSize() << " T/" << fobs[ifob]->l1tpftrackSize() << endl;
    LogTrace("TriggerSummaryProducerRaw")
        << "TriggerSummaryProducerRaw::addFilterObjects(   )"
        << "\n fobs[ifob]->l1tmuonIds().size() = " << fobs[ifob]->l1tmuonIds().size()
        << "\n fobs[ifob]->l1tmuonRefs().size() = " << fobs[ifob]->l1tmuonRefs().size() << endl;
    LogTrace("TriggerSummaryProducerRaw")
        << "TriggerSummaryProducerRaw::addFilterObjects(   )"
        << "\n fobs[ifob]->l1tegammaIds().size() = " << fobs[ifob]->l1tegammaIds().size()
        << "\n fobs[ifob]->l1tegammaRefs().size() = " << fobs[ifob]->l1tegammaRefs().size() << endl;
    LogTrace("TriggerSummaryProducerRaw")
        << "TriggerSummaryProducerRaw::addFilterObjects(   )"
        << "\n fobs[ifob]->l1tjetIds().size() = " << fobs[ifob]->l1tjetIds().size()
        << "\n fobs[ifob]->l1tjetRefs().size() = " << fobs[ifob]->l1tjetRefs().size() << endl;
    LogTrace("TriggerSummaryProducerRaw")
        << "TriggerSummaryProducerRaw::addFilterObjects(   )"
        << "\n fobs[ifob]->l1ttauIds().size() = " << fobs[ifob]->l1ttauIds().size()
        << "\n fobs[ifob]->l1ttauRefs().size() = " << fobs[ifob]->l1ttauRefs().size() << endl;
    LogTrace("TriggerSummaryProducerRaw")
        << "TriggerSummaryProducerRaw::addFilterObjects(   )"
        << "\n fobs[ifob]->l1tetsumIds().size() = " << fobs[ifob]->l1tetsumIds().size()
        << "\n fobs[ifob]->l1tetsumRefs().size() = " << fobs[ifob]->l1tetsumRefs().size() << endl;
    product.addFilterObject(tag, *fobs[ifob]);
  }

  // place product in Event
  OrphanHandle<TriggerEventWithRefs> ref = iEvent.emplace(putToken_, std::move(product));
  LogTrace("TriggerSummaryProducerRaw") << "Number of filter objects packed: " << ref->size();

  return;
}
