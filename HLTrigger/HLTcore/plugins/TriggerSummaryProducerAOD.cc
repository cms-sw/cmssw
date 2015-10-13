/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <algorithm>
#include <memory>
#include <typeinfo>

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerAOD.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps, const GlobalInputTags * gt) : 
  pn_(ps.getParameter<std::string>("processName")),
  filterTagsEvent_(pn_!="*"),
  filterTagsStream_(pn_!="*"),
  collectionTagsEvent_(pn_!="*"),
  collectionTagsStream_(pn_!="*"),
  toc_(),
  tags_(),
  offset_(),
  keys_(),
  ids_(),
  maskFilters_()
{
  if (pn_=="@") {
    edm::Service<edm::service::TriggerNamesService> tns;
    if (tns.isAvailable()) {
      pn_ = tns->getProcessName();
    } else {
      edm::LogError("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService not available!";
      pn_="*";
    }

    filterTagsEvent_     =InputTagSet(pn_!="*");
    filterTagsStream_    =InputTagSet(pn_!="*");
    collectionTagsEvent_ =InputTagSet(pn_!="*");
    collectionTagsStream_=InputTagSet(pn_!="*");
  }
  LogDebug("TriggerSummaryProducerAOD") << "Using process name: '" << pn_ <<"'";

  filterTagsStream_.clear();
  collectionTagsStream_.clear();

  produces<trigger::TriggerEvent>();

  getTriggerFilterObjectWithRefs_ = edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs>(edm::ProcessMatch(pn_), this);
  getRecoEcalCandidateCollection_ = edm::GetterOfProducts<reco::RecoEcalCandidateCollection>(edm::ProcessMatch(pn_), this);
  getElectronCollection_ = edm::GetterOfProducts<reco::ElectronCollection>(edm::ProcessMatch(pn_), this);
  getRecoChargedCandidateCollection_ = edm::GetterOfProducts<reco::RecoChargedCandidateCollection>(edm::ProcessMatch(pn_), this);
  getCaloJetCollection_ = edm::GetterOfProducts<reco::CaloJetCollection>(edm::ProcessMatch(pn_), this);
  getCompositeCandidateCollection_ = edm::GetterOfProducts<reco::CompositeCandidateCollection>(edm::ProcessMatch(pn_), this);
  getMETCollection_ = edm::GetterOfProducts<reco::METCollection>(edm::ProcessMatch(pn_), this);
  getCaloMETCollection_ = edm::GetterOfProducts<reco::CaloMETCollection>(edm::ProcessMatch(pn_), this);
  getIsolatedPixelTrackCandidateCollection_ = edm::GetterOfProducts<reco::IsolatedPixelTrackCandidateCollection>(edm::ProcessMatch(pn_), this);
  getL1EmParticleCollection_ = edm::GetterOfProducts<l1extra::L1EmParticleCollection>(edm::ProcessMatch(pn_), this);
  getL1MuonParticleCollection_ = edm::GetterOfProducts<l1extra::L1MuonParticleCollection>(edm::ProcessMatch(pn_), this);
  getL1JetParticleCollection_ = edm::GetterOfProducts<l1extra::L1JetParticleCollection>(edm::ProcessMatch(pn_), this);
  getL1EtMissParticleCollection_ = edm::GetterOfProducts<l1extra::L1EtMissParticleCollection>(edm::ProcessMatch(pn_), this);
  getL1HFRingsCollection_ = edm::GetterOfProducts<l1extra::L1HFRingsCollection>(edm::ProcessMatch(pn_), this);
  getPFJetCollection_ = edm::GetterOfProducts<reco::PFJetCollection>(edm::ProcessMatch(pn_), this);
  getPFTauCollection_ = edm::GetterOfProducts<reco::PFTauCollection>(edm::ProcessMatch(pn_), this);
  getPFMETCollection_ = edm::GetterOfProducts<reco::PFMETCollection>(edm::ProcessMatch(pn_), this);

  callWhenNewProductsRegistered([this](edm::BranchDescription const& bd){
    getTriggerFilterObjectWithRefs_(bd);
    getRecoEcalCandidateCollection_(bd);
    getElectronCollection_(bd);
    getRecoChargedCandidateCollection_(bd);
    getCaloJetCollection_(bd);
    getCompositeCandidateCollection_(bd);
    getMETCollection_(bd);
    getCaloMETCollection_(bd);
    getIsolatedPixelTrackCandidateCollection_(bd);
    getL1EmParticleCollection_(bd);
    getL1MuonParticleCollection_(bd);
    getL1JetParticleCollection_(bd);
    getL1EtMissParticleCollection_(bd);
    getL1HFRingsCollection_(bd);
    getPFJetCollection_(bd);
    getPFTauCollection_(bd);
    getPFMETCollection_(bd);
  });
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

void TriggerSummaryProducerAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName","@");
  descriptions.add("triggerSummaryProducerAOD", desc);
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

   std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs;
   getTriggerFilterObjectWithRefs_.fillHandles(iEvent, fobs);

   const unsigned int nfob(fobs.size());
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
     const vector<string>& collectionTags_(fobs[ifob]->getCollectionTagsAsStrings());
     const unsigned int ncol(collectionTags_.size());
     if (ncol>0) {
       nf++;
       maskFilters_[ifob]=true;
       const string& label    (fobs[ifob].provenance()->moduleLabel());
       const string& instance (fobs[ifob].provenance()->productInstanceName());
       const string& process  (fobs[ifob].provenance()->processName());
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
   collectionTagsStream_.insert(collectionTagsEvent_.begin(),collectionTagsEvent_.end());
   filterTagsStream_.insert(filterTagsEvent_.begin(),filterTagsEvent_.end());

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
   fillTriggerObjectCollections<          RecoEcalCandidateCollection>(iEvent, getRecoEcalCandidateCollection_);
   fillTriggerObjectCollections<                   ElectronCollection>(iEvent, getElectronCollection_);
   fillTriggerObjectCollections<       RecoChargedCandidateCollection>(iEvent, getRecoChargedCandidateCollection_);
   fillTriggerObjectCollections<                    CaloJetCollection>(iEvent, getCaloJetCollection_);
   fillTriggerObjectCollections<         CompositeCandidateCollection>(iEvent, getCompositeCandidateCollection_);
   fillTriggerObjectCollections<                        METCollection>(iEvent, getMETCollection_);
   fillTriggerObjectCollections<                    CaloMETCollection>(iEvent, getCaloMETCollection_);
   fillTriggerObjectCollections<IsolatedPixelTrackCandidateCollection>(iEvent, getIsolatedPixelTrackCandidateCollection_);
   ///
   fillTriggerObjectCollections<               L1EmParticleCollection>(iEvent, getL1EmParticleCollection_);
   fillTriggerObjectCollections<             L1MuonParticleCollection>(iEvent, getL1MuonParticleCollection_);
   fillTriggerObjectCollections<              L1JetParticleCollection>(iEvent, getL1JetParticleCollection_);
   fillTriggerObjectCollections<           L1EtMissParticleCollection>(iEvent, getL1EtMissParticleCollection_);
   fillTriggerObjectCollections<                  L1HFRingsCollection>(iEvent, getL1HFRingsCollection_);
   ///
   fillTriggerObjectCollections<                      PFJetCollection>(iEvent, getPFJetCollection_);
   fillTriggerObjectCollections<                      PFTauCollection>(iEvent, getPFTauCollection_);
   fillTriggerObjectCollections<                      PFMETCollection>(iEvent, getPFMETCollection_);
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
       const string& label    (fobs[ifob].provenance()->moduleLabel());
       const string& instance (fobs[ifob].provenance()->productInstanceName());
       const string& process  (fobs[ifob].provenance()->processName());
       const edm::InputTag filterTag(label,instance,process);
       ids_.clear();
       keys_.clear();
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->photonIds()   ,fobs[ifob]->photonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->electronIds() ,fobs[ifob]->electronRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->muonIds()     ,fobs[ifob]->muonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->jetIds()      ,fobs[ifob]->jetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->compositeIds(),fobs[ifob]->compositeRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->basemetIds()  ,fobs[ifob]->basemetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->calometIds()  ,fobs[ifob]->calometRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->pixtrackIds() ,fobs[ifob]->pixtrackRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->l1emIds()     ,fobs[ifob]->l1emRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->l1muonIds()   ,fobs[ifob]->l1muonRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->l1jetIds()    ,fobs[ifob]->l1jetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->l1etmissIds() ,fobs[ifob]->l1etmissRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->l1hfringsIds(),fobs[ifob]->l1hfringsRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->pfjetIds()    ,fobs[ifob]->pfjetRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->pftauIds()    ,fobs[ifob]->pftauRefs());
       fillFilterObjectMembers(iEvent,filterTag,fobs[ifob]->pfmetIds()    ,fobs[ifob]->pfmetRefs());
       product->addFilter(filterTag,ids_,keys_);
     }
   }

   OrphanHandle<TriggerEvent> ref = iEvent.put(product);
   LogTrace("TriggerSummaryProducerAOD") << "Number of physics objects packed: " << ref->sizeObjects();
   LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects packed: " << ref->sizeFilters();

}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjectCollections(const edm::Event& iEvent, edm::GetterOfProducts<C>& getter) {

  /// this routine accesses the original (L3) collections (with C++
  /// typename C), extracts 4-momentum and id of each collection
  /// member, and packs this up

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  vector<Handle<C> > collections;
  getter.fillHandles(iEvent, collections);
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

void TriggerSummaryProducerAOD::fillTriggerObject(const reco::PFMET& object) {

  using namespace reco;
  using namespace trigger;

  toc_.push_back( TriggerObject(object) );
  toc_.push_back(TriggerObject(TriggerTET    ,object.sumEt()         ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerMETSig ,object.mEtSig()        ,0.0,0.0,0.0));
  toc_.push_back(TriggerObject(TriggerELongit,object.e_longitudinal(),0.0,0.0,0.0));

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
    if (!(pid.isValid())) {
      LogError("TriggerSummaryProducerAOD")
	<< "Iinvalid pid: " << pid
	<< " FilterTag / Key: " << tag.encode()
	<< " / " << i << "of" << n
	<< " CollectionTag / Key: "
	<< " <Unrecoverable>"
	<< " / " << refs[i].key()
	<< " CollectionType: " << typeid(C).name();
    } else if (offset_.find(pid)==offset_.end()) {
      const string&    label(iEvent.getProvenance(pid).moduleLabel());
      const string& instance(iEvent.getProvenance(pid).productInstanceName());
      const string&  process(iEvent.getProvenance(pid).processName());
      LogError("TriggerSummaryProducerAOD")
	<< "Uunknown pid: " << pid
	<< " FilterTag / Key: " << tag.encode()
	<< " / " << i << "of" << n
	<< " CollectionTag / Key: "
	<< InputTag(label,instance,process).encode()
	<< " / " << refs[i].key()
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

void TriggerSummaryProducerAOD::fillFilterObjectMember(const int& offset, const int& id, const edm::Ref<reco::PFMETCollection> & ref) {

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

void TriggerSummaryProducerAOD::endStream() {
  globalCache()->collectionTagsGlobal_.insert(collectionTagsStream_.begin(),collectionTagsStream_.end());
  globalCache()->filterTagsGlobal_.insert(filterTagsStream_.begin(),filterTagsStream_.end());
  return;
}

void TriggerSummaryProducerAOD::globalEndJob(const GlobalInputTags * globalInputTags) {

  using namespace std;
  using namespace edm;
  using namespace trigger;

  LogVerbatim("TriggerSummaryProducerAOD") << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::globalEndJob - accumulated tags:" << endl;

  InputTagSet filterTags(false);
  InputTagSet collectionTags(false);

  filterTags.insert(globalInputTags->filterTagsGlobal_.begin(),globalInputTags->filterTagsGlobal_.end());
  collectionTags.insert(globalInputTags->collectionTagsGlobal_.begin(),globalInputTags->collectionTagsGlobal_.end());

  const unsigned int nc(collectionTags.size());
  const unsigned int nf(filterTags.size());
  LogVerbatim("TriggerSummaryProducerAOD") << " Overall number of Collections/Filters: "
		  << nc << "/" << nf << endl;

  LogVerbatim("TriggerSummaryProducerAOD") << " The collections: " << nc << endl;
  const InputTagSet::const_iterator cb(collectionTags.begin());
  const InputTagSet::const_iterator ce(collectionTags.end());
  for ( InputTagSet::const_iterator ci=cb; ci!=ce; ++ci) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(cb,ci) << " " << ci->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << " The filters:" << nf << endl;
  const InputTagSet::const_iterator fb(filterTags.begin());
  const InputTagSet::const_iterator fe(filterTags.end());
  for ( InputTagSet::const_iterator fi=fb; fi!=fe; ++fi) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(fb,fi) << " " << fi->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::endJob." << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << endl;

  return;

}
