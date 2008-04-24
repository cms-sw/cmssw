/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2008/04/24 10:28:08 $
 *  $Revision: 1.18 $
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

  bool cmpTagsLE(const edm::InputTag& a, const edm::InputTag& b) {
    return a.encode()<=b.encode();
  }

  bool cmpTagsEQ(const edm::InputTag& a, const edm::InputTag& b) {
    return a.encode()==b.encode();
  }

//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps) : 
  pn_(ps.getParameter<std::string>("processName")),
  selector_(edm::ProcessNameSelector(pn_)),
  tns_(),
  collectionTags_(),
  filterTags_(),
  toc_(),
  offset_(),
  fobs_(),
  keys_(),
  ids_(),
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
   /// Record those L3 filters and L3 collections.
   filterTags_.clear();
   collectionTags_.clear();
   vector<InputTag> collectionTags(0);
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     collectionTags.clear();
     fobs_[ifob]->getCollectionTags(collectionTags);
     if (collectionTags.size()>0) {
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());

       filterTags_.push_back(InputTag(label,instance,process));
       collectionTags_.insert(collectionTags_.end(),collectionTags.begin(),collectionTags.end());
     }
   }

   ///
   /// sort list of L3 collections and make entries unique
   sort (collectionTags_.begin(),collectionTags_.end(),cmpTagsLE);
   const trigger::size_type nc0(collectionTags_.size());
   LogTrace("") << "Number of collections requested " << nc0;
   std::cout    << "Number of collections requested " << nc0 << std::endl;
   for (trigger::size_type i=0; i!=nc0; ++i) {
     LogTrace("") << i << " " << collectionTags_[i].encode();
     //    std::cout    << i << " " << collectionTags_[i].encode() << std::endl;
   }
   collectionTags_.resize(distance(collectionTags_.begin(),unique(collectionTags_.begin(),collectionTags_.end(),cmpTagsEQ)));
   const trigger::size_type nc1(collectionTags_.size());
   LogTrace("") << "Number of unique collections requested " << nc1;
   std::cout    << "Number of unique collections requested " << nc1 << std::endl;
   for (trigger::size_type i=0; i!=nc1; ++i) {
     LogTrace("") << i << " " << collectionTags_[i].encode();
     std::cout    << i << " " << collectionTags_[i].encode() << std::endl;
   }

   ///
   /// sort list of L3 filters and make entries unique
   sort (filterTags_.begin(),filterTags_.end(),cmpTagsLE);
   const trigger::size_type nf0(filterTags_.size());
   LogTrace("") << "Number of filters requested " << nf0;
   std::cout    << "Number of filters requested " << nf0 << std::endl;
   for (trigger::size_type i=0; i!=nf0; ++i) {
     LogTrace("") << i << " " << filterTags_[i].encode();
     //    std::cout    << i << " " << filterTags_[i].encode() << std::endl;
   }
   filterTags_.resize(distance(filterTags_.begin(),unique(filterTags_.begin(),filterTags_.end(),cmpTagsEQ)));
   const trigger::size_type nf1(filterTags_.size());
   LogTrace("") << "Number of unique filters requested " << nf1;
   std::cout    << "Number of unique filters requested " << nf1 << std::endl;
   for (trigger::size_type i=0; i!=nf1; ++i) {
     LogTrace("") << i << " " << filterTags_[i].encode();
     std::cout    << i << " " << filterTags_[i].encode() << std::endl;
   }
   assert(nf0==nf1);

   ///
   /// Now the processing:
   /// first trigger objects from L3 collections, then L3 filter objects
   ///
   /// create trigger objects, fill triggerobjectcollection and offset map
   toc_.clear();
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
   const size_type nto(toc_.size());
   LogDebug("") << "Number of physics objects found: " << nto;

   ///
   /// find number of L3 filters and create mask
   size_type nfo(0);
   nfo = fillMask(fobs_,filterTags_);
   assert(nfo==filterTags_.size());

   ///
   /// construct single AOD product, reserving capacity
   auto_ptr<TriggerEvent> product(new TriggerEvent(pn_,nto,nfo));

   /// fill trigger object collection
   product->addObjects(toc_);

   /// fill the filter objects
   for (size_type ifob=0; ifob!=nfob; ++ifob) {
     if (mask_[ifob]) {
       const std::string& label(fobs_[ifob].provenance()->moduleLabel());
       ids_.clear();
       keys_.clear();
       fillFilterObjects(label,fobs_[ifob]->photonIds()   ,fobs_[ifob]->photonRefs());
       fillFilterObjects(label,fobs_[ifob]->electronIds() ,fobs_[ifob]->electronRefs());
       fillFilterObjects(label,fobs_[ifob]->muonIds()     ,fobs_[ifob]->muonRefs());
       fillFilterObjects(label,fobs_[ifob]->jetIds()      ,fobs_[ifob]->jetRefs());
       fillFilterObjects(label,fobs_[ifob]->compositeIds(),fobs_[ifob]->compositeRefs());
       fillFilterObjects(label,fobs_[ifob]->metIds()      ,fobs_[ifob]->metRefs());
       fillFilterObjects(label,fobs_[ifob]->htIds()       ,fobs_[ifob]->htRefs());
       fillFilterObjects(label,fobs_[ifob]->pixtrackIds() ,fobs_[ifob]->pixtrackRefs());
       fillFilterObjects(label,fobs_[ifob]->l1emIds()     ,fobs_[ifob]->l1emRefs());
       fillFilterObjects(label,fobs_[ifob]->l1muonIds()   ,fobs_[ifob]->l1muonRefs());
       fillFilterObjects(label,fobs_[ifob]->l1jetIds()    ,fobs_[ifob]->l1jetRefs());
       fillFilterObjects(label,fobs_[ifob]->l1etmissIds() ,fobs_[ifob]->l1etmissRefs());
       product->addFilter(label,ids_,keys_);
     }
   }

   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   LogTrace("") << "Number of physics objects packed: " << ref->sizeObjects();
   LogTrace("") << "Number of filter  objects packed: " << ref->sizeFilters();

}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjects(const edm::Event& iEvent) {

  /// this routine accesses the original (L3) collections (with C++
  /// typename C), extracts 4-momentum and id, and packs this up in a
  /// TriggerObjectCollection, i.e., a linearised vector of
  /// TriggerObjects

  using namespace std;
  using namespace edm;
  using namespace trigger;

  vector<Handle<C> > collections;
  iEvent.getMany(selector_,collections);
  const size_type nc(collections.size());

  fillMask(collections,collectionTags_);

  for (size_type ic=0; ic!=nc; ++ic) {
    if (mask_[ic]) {
      const ProductID pid(collections[ic].provenance()->productID());
      assert(offset_.find(pid)==offset_.end()); // else duplicate pid
      offset_[pid]=toc_.size();
      const size_type n(collections[ic]->size());
      for (size_type i=0; i!=n; ++i) {
	toc_.push_back(TriggerObject( (*collections[ic])[i] ));
      }
    }
  }

}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjects(const std::string& label, const trigger::Vids& ids, const std::vector<edm::Ref<C> >& refs) {

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
	   << " FilterLabel: " << label
	   << " CollectionType: " << typeid(C).name()
	   << endl;
    }
    assert(offset_.find(pid)!=offset_.end()); // else unknown pid
    keys_.push_back(offset_[pid]+refs[i].key());
    ids_.push_back(ids[i]);
  }

}

template <typename C>
trigger::size_type TriggerSummaryProducerAOD::fillMask(
  const std::vector<edm::Handle<C> >& products, 
  const std::vector<edm::InputTag>& wanted ) {

  /// this routine filles the mask of Boolean values for the list of
  /// products found in the Event, based on a list of wanted products
  /// specified by their InputTag given by the module configuration

  using namespace std;
  using namespace edm;
  using namespace trigger;

  const size_type np(products.size());
  const size_type nw(wanted.size());
  // LogTrace("") << np <<  " " << nw;

  mask_.clear();
  mask_.resize(np,false);
  
  size_type n(0);
  for (size_type ip=0; ip!=np; ++ip) {
    mask_[ip]=false;

    const string& label    (products[ip].provenance()->moduleLabel());
    const string& instance (products[ip].provenance()->productInstanceName());
    const string& process  (products[ip].provenance()->processName());

    // LogTrace("") << "MASK P: " << ip << " "+label+" "+instance+" "+process;

    for (size_type iw=0; iw!=nw; ++iw) {
      const string& tagLabel    (wanted[iw].label());
      const string& tagInstance (wanted[iw].instance());
      const string& tagProcess  (wanted[iw].process());
      // LogTrace("") << "MASK W: " << iw << " "+tagLabel+" "+tagInstance+" "+tagProcess;
      if (
 	   (label   ==tagLabel   ) &&
	   (instance==tagInstance) &&
	  ((process ==tagProcess )||(tagProcess=="")||(pn_=="*"))
	 ) {
	mask_[ip]=true;
        // LogTrace("") << "MASK match found!";
	++n;
	break;
      }
    }
    
  }

  return n;

}
