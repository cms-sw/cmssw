/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2008/05/02 13:29:56 $
 *  $Revision: 1.21 $
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
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

#include <algorithm>
#include <typeinfo>


//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName")),
  selector_(edm::ProcessNameSelector(pn_)),
  tns_(),
  collectionTags_(),
  collectionTagsEvent_(),
  collectionTagsGlobal_(),
  filterTagsEvent_(),
  filterTagsGlobal_(),
  toc_(),
  tags_(),
  offset_(),
  fobs_(),
  keys_(),
  ids_(),
  maskFilters_()
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

  collectionTagsGlobal_.clear();
  filterTagsGlobal_.clear();

  produces<trigger::TriggerEvent>();

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
   using namespace l1extra;
   using namespace trigger;

   ///
   /// get hold of filter objects
   fobs_.clear();
   iEvent.getMany(selector_,fobs_);
   const size_type nfob(fobs_.size());
   LogTrace("") << "Number of filter  objects found: " << nfob;

   ///
   /// check whether collection tags are recorded in filterobjects; if
   /// so, these are L3 collections to be packed up, and the
   /// corresponding filter is a L3 filter also to be packed up.
   /// Record the InputTags of those L3 filters and L3 collections.
   maskFilters_.clear();
   maskFilters_.resize(nfob,false);
   collectionTagsEvent_.clear();
   filterTagsEvent_.clear();
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     maskFilters_[ifob]=false;
     collectionTags_.clear();
     fobs_[ifob]->getCollectionTags(collectionTags_);
     if (collectionTags_.size()>0) {
       maskFilters_[ifob]=true;
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());
       filterTagsEvent_.insert(InputTag(label,instance,process));
       collectionTagsEvent_.insert(collectionTags_.begin(),collectionTags_.end());
     }
   }
   collectionTagsGlobal_.insert(collectionTagsEvent_.begin(),collectionTagsEvent_.end());
   filterTagsGlobal_.insert(filterTagsEvent_.begin(),filterTagsEvent_.end());

   ///
   const size_type nc(collectionTagsEvent_.size());
   LogTrace("") << "Number of unique collections requested " << nc;
   //   cout    << "Number of unique collections requested " << nc << endl;
   const InputTagSet::const_iterator cb(collectionTagsEvent_.begin());
   const InputTagSet::const_iterator ce(collectionTagsEvent_.end());
   for (InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
     LogTrace("") << distance(cb,ci) << " " << ci->encode();
     //   cout    << distance(cb,ci) << " " << ci->encode() << endl;
   }

   ///
   const size_type nf(filterTagsEvent_.size());
   LogTrace("") << "Number of unique filters requested " << nf;
   //   cout    << "Number of unique filters requested " << nf << endl;
   const InputTagSet::const_iterator fb(filterTagsEvent_.begin());
   const InputTagSet::const_iterator fe(filterTagsEvent_.end());
   for (InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
     LogTrace("") << distance(fb,fi) << " " << fi->encode();
     //   cout    << distance(fb,fi) << " " << fi->encode() << endl;
   }

   ///
   /// Now the processing:
   /// first trigger objects from L3 collections, then L3 filter objects
   ///
   /// create trigger objects, fill triggerobjectcollection and offset map
   toc_.clear();
   tags_.clear();
   keys_.clear();
   offset_.clear();
   fillTriggerObjects<          RecoEcalCandidateCollection>(iEvent);
   fillTriggerObjects<                   ElectronCollection>(iEvent);
   fillTriggerObjects<       RecoChargedCandidateCollection>(iEvent);
   fillTriggerObjects<                    CaloJetCollection>(iEvent);
   fillTriggerObjects<         CompositeCandidateCollection>(iEvent);
   fillTriggerObjects<                    CaloMETCollection>(iEvent);
   fillTriggerObjects<                        METCollection>(iEvent);
   fillTriggerObjects<IsolatedPixelTrackCandidateCollection>(iEvent);
   ///
   fillTriggerObjects<               L1EmParticleCollection>(iEvent);
   fillTriggerObjects<             L1MuonParticleCollection>(iEvent);
   fillTriggerObjects<              L1JetParticleCollection>(iEvent);
   fillTriggerObjects<           L1EtMissParticleCollection>(iEvent);
   ///
   const size_type nk(tags_.size());
   LogDebug("") << "Number of collections found: " << nk;
   const size_type no(toc_.size());
   LogDebug("") << "Number of physics objects found: " << no;

   ///
   /// construct single AOD product, reserving capacity
   auto_ptr<TriggerEvent> product(new TriggerEvent(pn_,nk,no,nf));

   /// fill trigger object collection
   product->addCollections(tags_,keys_);
   product->addObjects(toc_);

   /// fill the L3 filter objects
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     if (maskFilters_[ifob]) {
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());
       const InputTag filterTag(InputTag(label,instance,process));
       ids_.clear();
       keys_.clear();
       fillFilterObjects(filterTag,fobs_[ifob]->photonIds()   ,fobs_[ifob]->photonRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->electronIds() ,fobs_[ifob]->electronRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->muonIds()     ,fobs_[ifob]->muonRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->jetIds()      ,fobs_[ifob]->jetRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->compositeIds(),fobs_[ifob]->compositeRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->metIds()      ,fobs_[ifob]->metRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->htIds()       ,fobs_[ifob]->htRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->pixtrackIds() ,fobs_[ifob]->pixtrackRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->l1emIds()     ,fobs_[ifob]->l1emRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->l1muonIds()   ,fobs_[ifob]->l1muonRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->l1jetIds()    ,fobs_[ifob]->l1jetRefs());
       fillFilterObjects(filterTag,fobs_[ifob]->l1etmissIds() ,fobs_[ifob]->l1etmissRefs());
       product->addFilter(filterTag,ids_,keys_);
     }
   }

   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   LogTrace("") << "Number of physics objects packed: " << ref->sizeObjects();
   LogTrace("") << "Number of filter  objects packed: " << ref->sizeFilters();

}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjects(const edm::Event& iEvent) {

  /// this routine accesses the original (L3) collections (with C++
  /// typename C), extracts 4-momentum and id, and packs this up

  using namespace std;
  using namespace edm;
  using namespace trigger;

  vector<Handle<C> > collections;
  iEvent.getMany(selector_,collections);

  const size_type nc(collections.size());
  for (size_type ic=0; ic!=nc; ++ic) {
    const string& label    (collections[ic].provenance()->moduleLabel());
    const string& instance (collections[ic].provenance()->productInstanceName());
    const string& process  (collections[ic].provenance()->processName());
    const InputTag collectionTag(InputTag(label,instance,process));

    const InputTagSet::const_iterator tb(collectionTagsEvent_.begin());
    const InputTagSet::const_iterator te(collectionTagsEvent_.end());
    for (InputTagSet::const_iterator ti=tb; ti!=te; ++ti) {
      const string& tagLabel    (ti->label());
      const string& tagInstance (ti->instance());
      const string& tagProcess  (ti->process());
      if (
          (label   ==tagLabel   ) &&
          (instance==tagInstance) &&
          ((process ==tagProcess )||(tagProcess=="")||(pn_=="*"))
          ) {
	const ProductID pid(collections[ic].provenance()->productID());
	assert(offset_.find(pid)==offset_.end()); // else duplicate pid
	offset_[pid]=toc_.size();
	const size_type n(collections[ic]->size());
	for (size_type i=0; i!=n; ++i) {
	  toc_.push_back(TriggerObject( (*collections[ic])[i] ));
	}
	tags_.push_back(collectionTag);
	keys_.push_back(toc_.size());
	break;
      }
    } /// end loop over tags
  } /// end loop over handles
}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjects(const edm::InputTag& tag, const trigger::Vids& ids, const std::vector<edm::Ref<C> >& refs) {

  /// this routine takes a vector of Ref<C>s and determines the
  /// corresponding vector of keys (i.e., indices) into the
  /// TriggerObjectCollection

  using namespace std;
  using namespace edm;
  using namespace trigger;

  assert(ids.size()==refs.size());

  const size_type n(ids.size());
  for (size_type i=0; i!=n; ++i) {
    const ProductID pid(refs[i].id());
    if (offset_.find(pid)==offset_.end()) {
      offset_[pid]=0;
      cout << "#### Error in fillFilterObject (unknown pid):"
	   << " FilterTag: " << tag.encode()
	   << " CollectionType: " << typeid(C).name()
	   << endl;
    }
    assert(offset_.find(pid)!=offset_.end()); // else unknown pid
    keys_.push_back(offset_[pid]+refs[i].key());
    ids_.push_back(ids[i]);
  }

}

void TriggerSummaryProducerAOD::endJob() {

  using namespace std;
  using namespace edm;
  using namespace trigger;

  cout << "TriggerSummaryProducerAOD::endJob - accumulated tags:" << endl;

  const size_type nc(collectionTagsGlobal_.size());
  const size_type nf(filterTagsGlobal_.size());
  cout << " Overall number of Collections/Filters: "
       << nc << "/" << nf << endl;

  cout << " The collections:" << endl;
  const InputTagSet::const_iterator cb(collectionTagsGlobal_.begin());
  const InputTagSet::const_iterator ce(collectionTagsGlobal_.end());
  for (InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
    cout << "  " << distance(cb,ci) << " " << ci->encode() << endl;
  }

  cout << " The filters:" << endl;
  const InputTagSet::const_iterator fb(filterTagsGlobal_.begin());
  const InputTagSet::const_iterator fe(filterTagsGlobal_.end());
  for (InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
    cout << "  " << distance(fb,fi) << " " << fi->encode() << endl;
  }

  cout << "TriggerSummaryProducerAOD::endJob." << endl;

  return;

}
