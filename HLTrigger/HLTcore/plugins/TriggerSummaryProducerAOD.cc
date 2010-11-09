/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2010/11/08 15:47:45 $
 *  $Revision: 1.42 $
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
  filterTagsEvent_(pn_!="*"),
  filterTagsGlobal_(pn_!="*"),
  collectionTagsEvent_(pn_!="*"),
  collectionTagsGlobal_(pn_!="*"),
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
	edm::LogError("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService pointer = 0!";
	pn_="*";
      }
    } else {
      edm::LogError("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }
    selector_=edm::ProcessNameSelector(pn_);
    filterTagsEvent_     =InputTagSet(pn_!="*");
    filterTagsGlobal_    =InputTagSet(pn_!="*");
    collectionTagsEvent_ =InputTagSet(pn_!="*");
    collectionTagsGlobal_=InputTagSet(pn_!="*");
  }
  LogDebug("TriggerSummaryProducerAOD") << "Using process name: '" << pn_ <<"'";

  filterTagsGlobal_.clear();
  collectionTagsGlobal_.clear();

  produces<trigger::TriggerEvent>();

}

TriggerSummaryProducerAOD::~TriggerSummaryProducerAOD()
{
}

//
// member functions
//

namespace {
  inline void
  tokenizeTag(const std::string& tag, std::string& label, std::string& instance, std::string& process){
    
    using std::string;
    
    const char token(':');
    const string empty;
    
    label=tag;
    const string::size_type i1(label.find(token));
    if (i1==string::npos) {
      instance=empty;
      process=empty;
    } else {
      instance=label.substr(i1+1);
      label.resize(i1);
      const string::size_type i2(instance.find(token));
      if (i2==string::npos) {
	process=empty;
      } else {
	process=instance.substr(i2+1);
	instance.resize(i2);
      }
    }
  }
}

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
   const unsigned int nfob(fobs_.size());
   LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects found: " << nfob;

   string tagLabel,tagInstance,tagProcess;

   ///
   /// check whether collection tags are recorded in filterobjects; if
   /// so, these are L3 collections to be packed up, and the
   /// corresponding filter is a L3 filter also to be packed up.
   /// Record the InputTags of those L3 filters and L3 collections.
   maskFilters_.clear();
   maskFilters_.resize(nfob);
   filterTagsEvent_.clear();
   collectionTagsEvent_.clear();
   unsigned int nf(0);
   for (unsigned int ifob=0; ifob!=nfob; ++ifob) {
     maskFilters_[ifob]=false;
     const vector<string>& collectionTags_(fobs_[ifob]->getCollectionTagsAsStrings());
     const unsigned int ncol(collectionTags_.size());
     if (ncol>0) {
       nf++;
       maskFilters_[ifob]=true;
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());
       filterTagsEvent_.insert(InputTag(label,instance,process));
       for (unsigned int icol=0; icol!=ncol; ++icol) {
	 // overwrite process name (usually not set)
	 tokenizeTag(collectionTags_[icol],tagLabel,tagInstance,tagProcess);
	 collectionTagsEvent_.insert(InputTag(tagLabel,tagInstance,pn_));
       }
     }
   }
   /// check uniqueness count
   if (filterTagsEvent_.size()!=nf) {
     LogError("TriggerSummaryProducerAOD")
       << "Mismatch in number of filter tags: "
       << filterTagsEvent_.size() << "!=" << nf ;
   }

   /// accumulate for endJob printout
   collectionTagsGlobal_.insert(collectionTagsEvent_.begin(),collectionTagsEvent_.end());
   filterTagsGlobal_.insert(filterTagsEvent_.begin(),filterTagsEvent_.end());

   /// debug printout
   if (isDebugEnabled()) {

     /// event-by-event tags
     const unsigned int nc(collectionTagsEvent_.size());
     LogTrace("TriggerSummaryProducerAOD") << "Number of unique collections requested " << nc;
     const InputTagSet::const_iterator cb(collectionTagsEvent_.begin());
     const InputTagSet::const_iterator ce(collectionTagsEvent_.end());
     for ( InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
       LogTrace("TriggerSummaryProducerAOD") << distance(cb,ci) << " " << ci->encode();
     }
     const unsigned int nf(filterTagsEvent_.size());
     LogTrace("TriggerSummaryProducerAOD") << "Number of unique filters requested " << nf;
     const InputTagSet::const_iterator fb(filterTagsEvent_.begin());
     const InputTagSet::const_iterator fe(filterTagsEvent_.end());
     for ( InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
       LogTrace("TriggerSummaryProducerAOD") << distance(fb,fi) << " " << fi->encode();
     }

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
   const unsigned int nk(tags_.size());
   LogDebug("TriggerSummaryProducerAOD") << "Number of collections found: " << nk;
   const unsigned int no(toc_.size());
   LogDebug("TriggerSummaryProducerAOD") << "Number of physics objects found: " << no;

   ///
   /// construct single AOD product, reserving capacity
   auto_ptr<TriggerEvent> product(new TriggerEvent(pn_,nk,no,nf));

   /// fill trigger object collection
   product->addCollections(tags_,keys_);
   product->addObjects(toc_);

   /// fill the L3 filter objects
   for (unsigned int ifob=0; ifob!=nfob; ++ifob) {
     if (maskFilters_[ifob]) {
       const string& label    (fobs_[ifob].provenance()->moduleLabel());
       const string& instance (fobs_[ifob].provenance()->productInstanceName());
       const string& process  (fobs_[ifob].provenance()->processName());
       const edm::InputTag filterTag(label,instance,process);
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
  const unsigned int nc(collections.size());

  for (unsigned int ic=0; ic!=nc; ++ic) {
    const Provenance& provenance(*(collections[ic].provenance()));
    const string& label    (provenance.moduleLabel());
    const string& instance (provenance.productInstanceName());
    const string& process  (provenance.processName());
    const InputTag collectionTag(label,instance,process);

    if (collectionTagsEvent_.find(collectionTag)!=collectionTagsEvent_.end()) {
      const ProductID pid(collections[ic].provenance()->productID());
      if (offset_.find(pid)!=offset_.end()) {
	LogError("TriggerSummaryProducerAOD") << "Duplicate pid!";
      }
      offset_[pid]=toc_.size();
      const unsigned int n(collections[ic]->size());
      for (unsigned int i=0; i!=n; ++i) {
	fillTriggerObject( (*collections[ic])[i] );
      }
      tags_.push_back(collectionTag.encode());
      keys_.push_back(toc_.size());
    }

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

  if (ids.size()!=refs.size()) {
    LogError("TriggerSummaryProducerAOD") << "Vector length is different: "
					  << ids.size() << " " << refs.size();
  }

  const unsigned int n(min(ids.size(),refs.size()));
  for (unsigned int i=0; i!=n; ++i) {
    const ProductID pid(refs[i].id());
    if (offset_.find(pid)==offset_.end()) {
      const string&    label(iEvent.getProvenance(pid).moduleLabel());
      const string& instance(iEvent.getProvenance(pid).productInstanceName());
      const string&  process(iEvent.getProvenance(pid).processName());
      LogError("TriggerSummaryProducerAOD")
	<< "Uunknown pid:"
	<< " FilterTag/Key: " << tag.encode()
	<< "/" << i
	<< " CollectionTag/Key: "
	<< InputTag(label,instance,process).encode()
	<< "/" << refs[i].key()
	<< " CollectionType: " << typeid(C).name();
    } else {
      fillFilterObjectMember(offset_[pid],ids[i],refs[i]);
    }
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

  const unsigned int nc(collectionTagsGlobal_.size());
  const unsigned int nf(filterTagsGlobal_.size());
  LogVerbatim("TriggerSummaryProducerAOD") << " Overall number of Collections/Filters: "
		  << nc << "/" << nf << endl;

  LogVerbatim("TriggerSummaryProducerAOD") << " The collections: " << nc << endl;
  const InputTagSet::const_iterator cb(collectionTagsGlobal_.begin());
  const InputTagSet::const_iterator ce(collectionTagsGlobal_.end());
  for ( InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(cb,ci) << " " << ci->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << " The filters:" << nf << endl;
  const InputTagSet::const_iterator fb(filterTagsGlobal_.begin());
  const InputTagSet::const_iterator fe(filterTagsGlobal_.end());
  for ( InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(fb,fi) << " " << fi->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::endJob." << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << endl;

  return;

}
