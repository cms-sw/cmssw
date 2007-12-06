/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2007/12/06 08:27:31 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"

//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName")),
  selector_(edm::ProcessNameSelector(pn_)),
  tns_(),
  collectionTags_(ps.getParameter<std::vector<edm::InputTag> >("collections")),
  filterTags_(ps.getParameter<std::vector<edm::InputTag> >("filters")),
  offset_(),
  fobs_(),
  toc_(),
  keys_(),
  mask_()
{
  if (pn_=="@") {
    // use tns
    if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
      // get tns pointer
      tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
      if (tns_!=0) {
	pn_=tns_->getProcessName();
      } else {
	LogDebug("") << "HLT Error: TriggerNamesService pointer = 0!";
	pn_="*";
      }
    } else {
      LogDebug("") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }
    selector_=edm::ProcessNameSelector(pn_);
  }

  LogDebug("") << "Using process name: '" << pn_ <<"'";
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

   /// create trigger objects, fill triggerobjectcollection and offset map
   toc_.clear();
   offset_.clear();
   fillTriggerObjects<   RecoEcalCandidateCollection>(iEvent);
   fillTriggerObjects<            ElectronCollection>(iEvent);
   fillTriggerObjects<RecoChargedCandidateCollection>(iEvent);
   fillTriggerObjects<             CaloJetCollection>(iEvent);
   fillTriggerObjects<  CompositeCandidateCollection>(iEvent);
   fillTriggerObjects<             CaloMETCollection>(iEvent);
   fillTriggerObjects<                 METCollection>(iEvent);
   const size_type nto(toc_.size());
   LogDebug("") << "Number of physics objects found: " << nto;

   /// get hold of filter objects
   fobs_.clear();
   iEvent.getMany(selector_,fobs_);
   const size_type nfob(fobs_.size());
   LogDebug("") << "Number of filter  objects found: " << nfob;
   size_type nfo(0);
   nfo = fillMask(fobs_,filterTags_);

   /// construct single AOD product, reserving capacity
   auto_ptr<TriggerEvent> product(new TriggerEvent(nto,nfo));


   /// fill trigger object collection
   product->addObjects(toc_);

   /// fill the filter objects
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     if (mask_[ifob]) {
       keys_.clear();
       fillFilterKeys(fobs_[ifob]->photonRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->electronRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->muonRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->jetRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->compositeRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->metRefs(),keys_);
       fillFilterKeys(fobs_[ifob]->htRefs(),keys_);
       product->addFilter(fobs_[ifob].provenance()->moduleLabel(),keys_);
     }
   }

   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   LogTrace("") << "Number of physics objects packed: " << ref->sizeObjects();
   LogTrace("") << "Number of filter  objects packed: " << ref->sizeFilters();

}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjects(const edm::Event& iEvent) {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  vector<Handle<C> > collections;
  iEvent.getMany(selector_,collections);
  const size_type nc(collections.size());
  
  size_type no(0);
  no=fillMask(collections,collectionTags_);

  TriggerObjectCollection toc();
  toc_.reserve(no);
  
  no=0;
  for (size_type ic=0; ic!=nc; ++ic) {
    if (mask_[ic]) {
      const ProductID pid(collections[ic].provenance()->productID());
      assert(offset_.find(pid)==offset_.end());
      offset_[pid]=no;
      const size_type n(collections[ic]->size());
      for (size_type i=0; i!=n; ++i) {
	toc_.push_back(TriggerObject( (*collections[ic])[i] ));
      }
    }
  }
}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterKeys(const std::vector<edm::Ref<C> >& refs, trigger::Keys& keys) {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  const size_type n(refs.size());
  for (size_type i=0; i!=n; ++i) {
    const ProductID pid(refs[i].id());
    assert(offset_.find(pid)!=offset_.end());
    keys.push_back(offset_[pid]+refs[i].key());
  }
}

template <typename C>
trigger::size_type TriggerSummaryProducerAOD::fillMask(
  const std::vector<edm::Handle<C> >& products, 
  const std::vector<edm::InputTag>& wanted ) {

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  const size_type np(products.size());
  const size_type nw(wanted.size());

  mask_.clear();
  mask_.resize(np,false);
  
  size_type n(0);
  for (size_type ip=0; ip!=np; ++ip) {
    mask_[ip]=false;

    const string& label    (products[ip].provenance()->moduleLabel());
    const string& instance (products[ip].provenance()->productInstanceName());
    const string& process  (products[ip].provenance()->processName());

    for (size_type iw=0; iw!=nw; ++iw) {
      if ( (label   ==collectionTags_[iw].label()   ) &&
	   (instance==collectionTags_[iw].instance()) &&
	   (process ==collectionTags_[iw].process() ) ) {
	mask_[ip]=true;
	++n;
	break;
      }
    }
    
  }

  return n;

}
