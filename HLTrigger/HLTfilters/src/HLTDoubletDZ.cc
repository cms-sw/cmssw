#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLTDoubletDZ.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"

#include<cmath>

//
// constructors and destructor
//
template<typename T1, typename T2>
HLTDoubletDZ<T1,T2>::HLTDoubletDZ(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  originTag1_(iConfig.template getParameter<std::vector<edm::InputTag> >("originTag1")),
  originTag2_(iConfig.template getParameter<std::vector<edm::InputTag> >("originTag2")),
  inputTag1_(iConfig.template getParameter<edm::InputTag>("inputTag1")),
  inputTag2_(iConfig.template getParameter<edm::InputTag>("inputTag2")),
  inputToken1_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag1_)),
  inputToken2_(consumes<trigger::TriggerFilterObjectWithRefs>(inputTag2_)),
  triggerType1_(iConfig.template getParameter<int>("triggerType1")),
  triggerType2_(iConfig.template getParameter<int>("triggerType2")),
  minDR_ (iConfig.template getParameter<double>("MinDR")),
  maxDZ_ (iConfig.template getParameter<double>("MaxDZ")),
  min_N_    (iConfig.template getParameter<int>("MinN")),
  checkSC_  (iConfig.template getParameter<bool>("checkSC")),
  same_     (inputTag1_.encode()==inputTag2_.encode())      // same collections to be compared?
{
}

template<typename T1, typename T2>
HLTDoubletDZ<T1,T2>::~HLTDoubletDZ()
{
}

template<typename T1, typename T2>
void
HLTDoubletDZ<T1,T2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  std::vector<edm::InputTag> originTag1(1,edm::InputTag("hltOriginal1"));
  std::vector<edm::InputTag> originTag2(1,edm::InputTag("hltOriginal2"));
  desc.add<std::vector<edm::InputTag> >("originTag1",originTag1);
  desc.add<std::vector<edm::InputTag> >("originTag2",originTag2);
  desc.add<edm::InputTag>("inputTag1",edm::InputTag("hltFiltered1"));
  desc.add<edm::InputTag>("inputTag2",edm::InputTag("hltFiltered2"));
  desc.add<int>("triggerType1",0);
  desc.add<int>("triggerType2",0);
  desc.add<double>("MinDR",-1.0);
  desc.add<double>("MaxDZ",0.2);
  desc.add<bool>("checkSC",false);
  desc.add<int>("MinN",1);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTDoubletDZ<T1,T2>).name()),desc);
}

// ------------ method called to produce the data  ------------
template<typename T1, typename T2>
bool
HLTDoubletDZ<T1,T2>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   bool accept(false);

   LogVerbatim("HLTDoubletDZ") << " XXX " << moduleLabel() << " 0 " << std::endl;

   std::vector<T1Ref> coll1;
   std::vector<T2Ref> coll2;

   // get hold of pre-filtered object collections
   Handle<TriggerFilterObjectWithRefs> handle1,handle2;
   if (iEvent.getByToken(inputToken1_, handle1) and iEvent.getByToken(inputToken2_, handle2)) {
     handle1->getObjects(triggerType1_, coll1);
     handle2->getObjects(triggerType2_, coll2);
     const size_type n1(coll1.size());
     const size_type n2(coll2.size());

     if (saveTags()) {
       InputTag tagOld;
       for (unsigned int i=0; i<originTag1_.size(); ++i) {
	 filterproduct.addCollectionTag(originTag1_[i]);
	 LogVerbatim("HLTDoubletDZ") << " XXX " << moduleLabel() << " 1a/" << i << " " << originTag1_[i].encode() << std::endl;
       }
       tagOld=InputTag();
       for (size_type i1=0; i1!=n1; ++i1) {
	 const ProductID pid(coll1[i1].id());
	 const string&    label(iEvent.getProvenance(pid).moduleLabel());
	 const string& instance(iEvent.getProvenance(pid).productInstanceName());
	 const string&  process(iEvent.getProvenance(pid).processName());
	 InputTag tagNew(InputTag(label,instance,process));
	 if (tagOld.encode()!=tagNew.encode()) {
	   filterproduct.addCollectionTag(tagNew);
	   tagOld=tagNew;
           LogVerbatim("HLTDoubletDZ") << " XXX " << moduleLabel() << " 1b " << tagNew.encode() << std::endl;
	 }
       }
       for (unsigned int i=0; i<originTag2_.size(); ++i) {
	 filterproduct.addCollectionTag(originTag2_[i]);
	 LogVerbatim("HLTDoubletDZ") << " XXX " << moduleLabel() << " 2a/" << originTag2_[i].encode() << std::endl;
       }
       tagOld=InputTag();
       for (size_type i2=0; i2!=n2; ++i2) {
	 const ProductID pid(coll2[i2].id());
	 const string&    label(iEvent.getProvenance(pid).moduleLabel());
	 const string& instance(iEvent.getProvenance(pid).productInstanceName());
	 const string&  process(iEvent.getProvenance(pid).processName());
	 InputTag tagNew(InputTag(label,instance,process));
	 if (tagOld.encode()!=tagNew.encode()) {
	   filterproduct.addCollectionTag(tagNew);
	   tagOld=tagNew;
           LogVerbatim("HLTDoubletDZ") << " XXX " << moduleLabel() << " 2b " << tagNew.encode() << std::endl;
	 }
       }
     }

     int n(0);
     T1Ref r1;
     T2Ref r2;
     Particle::LorentzVector p1,p2,p;
     for (unsigned int i1=0; i1!=n1; i1++) {
       r1=coll1[i1];
       const reco::Candidate& candidate1(*r1);
       unsigned int I(0);
       if (same_) {I=i1+1;}
       for (unsigned int i2=I; i2!=n2; i2++) {
	 r2=coll2[i2];
	 if (checkSC_) {
	   if (r1->superCluster().isNonnull() && r2->superCluster().isNonnull()) {
	     if (r1->superCluster() == r2->superCluster()) continue;
	   }
	 }
 	 const reco::Candidate& candidate2(*r2);
	 if ( reco::deltaR(candidate1, candidate2) < minDR_ ) continue;
	 if ( std::abs(candidate1.vz()-candidate2.vz()) > maxDZ_ ) continue;
	 n++;
	 filterproduct.addObject(triggerType1_,r1);
	 filterproduct.addObject(triggerType2_,r2);
       }
     }
     // filter decision
     accept = accept || (n>=min_N_);
   }

   return accept;
}
