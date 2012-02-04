/** \class HLTJetSortedVBFFilter
 *
 * See header file for documentation
 *
 *  $Date: 2012/02/03 16:37:51 $


 *  $Revision: 1.1 $
 *
 *  \author Jacopo Bernardini
 *
 */


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/JetMET/interface/HLTJetSortedVBFFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include<vector>
#include<string>
#include<typeinfo>

using namespace std;

//
// constructors and destructor//
//
template<typename T, int Tid>
HLTJetSortedVBFFilter<T,Tid>::HLTJetSortedVBFFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
 ,inputJets_   (iConfig.getParameter<edm::InputTag>("inputJets"   ))
 ,inputJetTags_(iConfig.getParameter<edm::InputTag>("inputJetTags"))
 ,mqq_         (iConfig.getParameter<double>       ("Mqq"         ))
 ,detaqq_      (iConfig.getParameter<double>       ("Detaqq"      ))
 ,detabb_      (iConfig.getParameter<double>       ("Detabb"      ))
 ,ptsqq_       (iConfig.getParameter<double>       ("Ptsumqq"     ))
 ,ptsbb_       (iConfig.getParameter<double>       ("Ptsumbb"     ))
 ,seta_        (iConfig.getParameter<double>       ("Etaq1Etaq2"  ))
 ,value_       (iConfig.getParameter<std::string>  ("value"       ))
{
}


template<typename T, int Tid>
HLTJetSortedVBFFilter<T,Tid>::~HLTJetSortedVBFFilter()
{ }

template<typename T, int Tid>
void
HLTJetSortedVBFFilter<T,Tid>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJets",edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("inputJetTags",edm::InputTag(""));
  desc.add<double>("Mqq",200);
  desc.add<double>("Detaqq",2.5);
  desc.add<double>("Detabb",10.);
  desc.add<double>("Ptsumqq",0.);
  desc.add<double>("Ptsumbb",0.);
  desc.add<double>("Etaq1Etaq2",40.);
  desc.add<std::string>("value","second");
  descriptions.add(string("hlt")+string(typeid(HLTJetSortedVBFFilter<T,Tid>).name()),desc);
}


//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T, int Tid>
bool 
HLTJetSortedVBFFilter<T,Tid>::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs& filterproduct)
{

   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   typedef vector<T> TCollection;
   typedef Ref<TCollection> TRef;
   vector<TRef> jetRefs(4);
     
   bool accept(false);

   if (saveTags()) filterproduct.addCollectionTag(inputJets_);

   vector<Jpair> Sorted;
   Sorted.clear();

   Handle<TCollection> jets;
   event.getByLabel(inputJets_,jets);
   Handle<JetTagCollection> jetTags;

   int nJet=0;
   double value(0.0);

   if (inputJetTags_.encode()=="") {
     if (jets->size()<4) return false;
     for (typename TCollection::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet) {
       if (value_=="Pt") {
	 value=jet->pt();
       } else if (value_=="Eta") {
	 value=jet->eta();
       } else if (value_=="Phi") {
	 value=jet->phi();
       } else {
	 value = 0.0;
       }
       Sorted.push_back(make_pair(value,nJet));
       ++nJet;
     }
     sort(Sorted.begin(),Sorted.end(),comparator);
     for (unsigned int i=0; i<4; ++i) {
       jetRefs[i]=TRef(jets,Sorted[i].second);
     }
   } else {
     event.getByLabel(inputJetTags_,jetTags);
     if (jetTags->size()<4) return false;
     for (JetTagCollection::const_iterator jet = jetTags->begin(); jet!=jetTags->end(); ++jet) {
       value = jet->second;
       Sorted.push_back(make_pair(value,nJet));
       ++nJet;
     }
     sort(Sorted.begin(),Sorted.end(),comparator);
     for (unsigned int i=0; i<4; ++i) {
       jetRefs[i]= TRef(jets,(*jetTags)[Sorted[i].second].first.key());
     }
   }

   Particle::LorentzVector b1,b2,q1,q2;
   b1 = jetRefs[3]->p4();
   b2 = jetRefs[2]->p4();
   q1 = jetRefs[1]->p4();
   q2 = jetRefs[0]->p4();

   double mqq_bs     = (q1+q2).M();
   double deltaetaqq = std::abs(q1.Eta()-q2.Eta());
   double deltaetabb = std::abs(b1.Eta()-b2.Eta());
   double ptsqq_bs   = (q1+q2).Pt();
   double ptsbb_bs   = (b1+b2).Pt();
   double signeta    = q1.Eta()*q2.Eta();

   if ( 
	(mqq_bs     > mqq_    ) &&
	(deltaetaqq > detaqq_ ) &&
	(deltaetabb > detabb_ ) &&
	(ptsqq_bs   > ptsqq_  ) &&
	(ptsbb_bs   > ptsbb_  ) &&
	(signeta    > seta_   )
	) {
     accept=true;
     for (unsigned int i=0; i<4; ++i) {
       filterproduct.addObject(static_cast<trigger::TriggerObjectType>(Tid),jetRefs[i]);
     }
   }

   return accept;
}
