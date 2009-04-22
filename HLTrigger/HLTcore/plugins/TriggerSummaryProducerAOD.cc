/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2009/04/08 16:39:25 $
 *  $Revision: 1.34 $
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
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "DataFormats/L1Trigger/interface/L1HFRings.h"
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
	LogDebug("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService pointer = 0!";
	pn_="*";
      }
    } else {
      LogDebug("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }
    selector_=edm::ProcessNameSelector(pn_);
  }
  LogDebug("TriggerSummaryProducerAOD") << "Using process name: '" << pn_ <<"'";

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
   LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects found: " << nfob;

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
     const size_type ncol(collectionTags_.size());
     if (ncol>0) {
       maskFilters_[ifob]=true;
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());
       filterTagsEvent_.insert(InputTag(label,instance,process));

       for (size_type icol=0; icol!=ncol; ++icol) {
	 const string&    label(collectionTags_[icol].label());
	 const string& instance(collectionTags_[icol].instance());
	 collectionTagsEvent_.insert(InputTag(label,instance,pn_));
       }
     }
   }
   collectionTagsGlobal_.insert(collectionTagsEvent_.begin(),collectionTagsEvent_.end());
   filterTagsGlobal_.insert(filterTagsEvent_.begin(),filterTagsEvent_.end());

   ///
   const size_type nc(collectionTagsEvent_.size());
   LogTrace("TriggerSummaryProducerAOD") << "Number of unique collections requested " << nc;
   //   cout    << "Number of unique collections requested " << nc << endl;
   const InputTagSet::const_iterator cb(collectionTagsEvent_.begin());
   const InputTagSet::const_iterator ce(collectionTagsEvent_.end());
   for (InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
     LogTrace("TriggerSummaryProducerAOD") << distance(cb,ci) << " " << ci->encode();
     //   cout    << distance(cb,ci) << " " << ci->encode() << endl;
   }

   ///
   const size_type nf(filterTagsEvent_.size());
   LogTrace("TriggerSummaryProducerAOD") << "Number of unique filters requested " << nf;
   //   cout    << "Number of unique filters requested " << nf << endl;
   const InputTagSet::const_iterator fb(filterTagsEvent_.begin());
   const InputTagSet::const_iterator fe(filterTagsEvent_.end());
   for (InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
     LogTrace("TriggerSummaryProducerAOD") << distance(fb,fi) << " " << fi->encode();
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
   fillTriggerObjectCollections<          RecoEcalCandidateCollection>(iEvent);
   fillTriggerObjectCollections<                   ElectronCollection>(iEvent);
   fillTriggerObjectCollections<       RecoChargedCandidateCollection>(iEvent);
   fillTriggerObjectCollections<                    CaloJetCollection>(iEvent);
   fillTriggerObjectCollections<         CompositeCandidateCollection>(iEvent);
   fillTriggerObjectCollections<                        METCollection>(iEvent);
   fillTriggerObjectCollections<                    CaloMETCollection>(iEvent);
   fillTriggerObjectCollections<IsolatedPixelTrackCandidateCollection>(iEvent);
   ///
   fillTriggerObjectCollections<               L1EmParticleCollection>(iEvent);
   fillTriggerObjectCollections<             L1MuonParticleCollection>(iEvent);
   fillTriggerObjectCollections<              L1JetParticleCollection>(iEvent);
   fillTriggerObjectCollections<           L1EtMissParticleCollection>(iEvent);
   fillTriggerObjectCollections<                  L1HFRingsCollection>(iEvent);
   ///
   const size_type nk(tags_.size());
   LogDebug("TriggerSummaryProducerAOD") << "Number of collections found: " << nk;
   const size_type no(toc_.size());
   LogDebug("TriggerSummaryProducerAOD") << "Number of physics objects found: " << no;

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
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->photonIds()   ,fobs_[ifob]->photonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->electronIds() ,fobs_[ifob]->electronRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->muonIds()     ,fobs_[ifob]->muonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->jetIds()      ,fobs_[ifob]->jetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->compositeIds(),fobs_[ifob]->compositeRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->basemetIds()  ,fobs_[ifob]->basemetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->calometIds()  ,fobs_[ifob]->calometRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->pixtrackIds() ,fobs_[ifob]->pixtrackRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->l1emIds()     ,fobs_[ifob]->l1emRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->l1muonIds()   ,fobs_[ifob]->l1muonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->l1jetIds()    ,fobs_[ifob]->l1jetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->l1etmissIds() ,fobs_[ifob]->l1etmissRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs_[ifob]->l1hfringsIds(),fobs_[ifob]->l1hfringsRefs());
       product->addFilter(filterTag,ids_,keys_);
     }
   }

   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   LogTrace("TriggerSummaryProducerAOD") << "Number of physics objects packed: " << ref->sizeObjects();
   LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects packed: " << ref->sizeFilters();

}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjectCollections(const edm::Event& iEvent) {

  /// this routine accesses the original (L3) collections (with C++
  /// typename C), extracts 4-momentum and id of each collection
  /// member, and packs this up

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
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
	  fillTriggerObject( (*collections[ic])[i] );
	}
	tags_.push_back(collectionTag);
	keys_.push_back(toc_.size());
	break;
      }
    } /// end loop over tags
  } /// end loop over handles
}

template <typename T>
void TriggerSummaryProducerAOD::fillTriggerObject(const T& object) {

  using namespace trigger;
  toc_.push_back( TriggerObject(object) );

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(const l1extra::L1HFRings& object) {

  using namespace l1extra;
  using namespace trigger;

  toc_.push_back(TriggerObject(TriggerL1HfRingEtSums,
       object.hfEtSum(L1HFRings::kRing1PosEta),
       object.hfEtSum(L1HFRings::kRing1NegEta),
       object.hfEtSum(L1HFRings::kRing2PosEta),
       object.hfEtSum(L1HFRings::kRing2NegEta) ) );
  toc_.push_back(TriggerObject(TriggerL1HfBitCounts,
       object.hfBitCount(L1HFRings::kRing1PosEta),
       object.hfBitCount(L1HFRings::kRing1NegEta),
       object.hfBitCount(L1HFRings::kRing2PosEta),
       object.hfBitCount(L1HFRings::kRing2NegEta) ) );

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(const l1extra::L1EtMissParticle& object) {

  using namespace l1extra;
  using namespace trigger;

  toc_.push_back( TriggerObject(object) );
  if (object.type()==L1EtMissParticle::kMET) {
    toc_.push_back(TriggerObject(TriggerL1ETT,object.etTotal(),0.0,0.0,0.0));
  } else if (object.type()==L1EtMissParticle::kMHT) {
    toc_.push_back(TriggerObject(TriggerL1HTT,object.etTotal(),0.0,0.0,0.0));
  } else {
    toc_.push_back(TriggerObject(0,           object.etTotal(),0.0,0.0,0.0));
  }

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(const reco::CaloMET& object) {

  using namespace reco;
  using namespace trigger;

  toc_.push_back( TriggerObject(object) );
  toc_.push_back(TriggerObject(TriggerTET    ,object.sumEt()         ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerMETSig ,object.mEtSig()        ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerELongit,object.e_longitudinal(),0.0,0.0,0.0));

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(const reco::MET& object) {

  using namespace reco;
  using namespace trigger;

  toc_.push_back( TriggerObject(object) );
  toc_.push_back(TriggerObject(TriggerTHT    ,object.sumEt()         ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerMHTSig ,object.mEtSig()        ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerHLongit,object.e_longitudinal(),0.0,0.0,0.0));

  return;
}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjectMembers(const edm::Event& iEvent, const edm::InputTag& tag, const trigger::Vids& ids, const std::vector<edm::Ref<C> >& refs) {

  /// this routine takes a vector of Ref<C>s and determines the
  /// corresponding vector of keys (i.e., indices) into the
  /// TriggerObjectCollection

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  assert(ids.size()==refs.size());

  const size_type n(ids.size());
  for (size_type i=0; i!=n; ++i) {
    const ProductID pid(refs[i].id());
    if (offset_.find(pid)==offset_.end()) {
      offset_[pid]=0;
      const string&    label(iEvent.getProvenance(pid).moduleLabel());
      const string& instance(iEvent.getProvenance(pid).productInstanceName());
      const string&  process(iEvent.getProvenance(pid).processName());
      cout << "#### Error in TriggerSummaryProducerAOD::fillFilterObject (unknown pid):"
	   << " FilterTag/Key: " << tag.encode()
	   << "/" << i
	   << " CollectionTag/Key: "
	   << InputTag(label,instance,process).encode()
	   << "/" << refs[i].key()
	   << " CollectionType: " << typeid(C).name()
	   << endl;
    }
    assert(offset_.find(pid)!=offset_.end()); // else unknown pid

    fillFilterObjectMember(offset_[pid],ids[i],refs[i]);
  }
  return;

}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<C> & ref) {

  keys_.push_back(offset+ref.key());
  ids_.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<l1extra::L1HFRingsCollection> & ref) {

  using namespace trigger;

  if (id==TriggerL1HfBitCounts) {
    keys_.push_back(offset+2*ref.key()+1);
  } else { // if (ids[i]==TriggerL1HfRingEtSums) {
    keys_.push_back(offset+2*ref.key()+0);
  }
  ids_.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<l1extra::L1EtMissParticleCollection> & ref) {

  using namespace trigger;

  if ( (id==TriggerL1ETT) || (id==TriggerL1HTT) ) {
    keys_.push_back(offset+2*ref.key()+1);
  } else {
    keys_.push_back(offset+2*ref.key()+0);
  }
  ids_.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<reco::CaloMETCollection> & ref) {

  using namespace trigger;

  if ( (id==TriggerTHT) || (id==TriggerTET) ) {
    keys_.push_back(offset+4*ref.key()+1);
  } else if ( (id==TriggerMETSig) || (id==TriggerMHTSig) ) {
    keys_.push_back(offset+4*ref.key()+2);
  } else if ( (id==TriggerELongit) || (id==TriggerHLongit) ) {
    keys_.push_back(offset+4*ref.key()+3);
  } else {
    keys_.push_back(offset+4*ref.key()+0);
  }
  ids_.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<reco::METCollection> & ref) {

  using namespace trigger;

  if ( (id==TriggerTHT) || (id==TriggerTET) ) {
    keys_.push_back(offset+4*ref.key()+1);
  } else if ( (id==TriggerMETSig) || (id==TriggerMHTSig) ) {
    keys_.push_back(offset+4*ref.key()+2);
  } else if ( (id==TriggerELongit) || (id==TriggerHLongit) ) {
    keys_.push_back(offset+4*ref.key()+3);
  } else {
    keys_.push_back(offset+4*ref.key()+0);
  }
  ids_.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::endJob() {

  using namespace std;
  using namespace edm;
  using namespace trigger;

  LogVerbatim("TriggerSummaryProducerAOD") << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::endJob - accumulated tags:" << endl;

  const size_type nc(collectionTagsGlobal_.size());
  const size_type nf(filterTagsGlobal_.size());
  LogVerbatim("TriggerSummaryProducerAOD") << " Overall number of Collections/Filters: "
		  << nc << "/" << nf << endl;

  LogVerbatim("TriggerSummaryProducerAOD") << " The collections:" << endl;
  const InputTagSet::const_iterator cb(collectionTagsGlobal_.begin());
  const InputTagSet::const_iterator ce(collectionTagsGlobal_.end());
  for (InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(cb,ci) << " " << ci->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << " The filters:" << endl;
  const InputTagSet::const_iterator fb(filterTagsGlobal_.begin());
  const InputTagSet::const_iterator fe(filterTagsGlobal_.end());
  for (InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(fb,fi) << " " << fi->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::endJob." << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << endl;

  return;

}
