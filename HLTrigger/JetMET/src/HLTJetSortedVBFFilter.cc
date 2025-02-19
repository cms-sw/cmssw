/** \class HLTJetSortedVBFFilter
 *
 * See header file for documentation
 *
 *  $Date: 2012/02/12 12:21:02 $


 *  $Revision: 1.6 $
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
template<typename T>
HLTJetSortedVBFFilter<T>::HLTJetSortedVBFFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
 ,inputJets_   (iConfig.getParameter<edm::InputTag>("inputJets"   ))
 ,inputJetTags_(iConfig.getParameter<edm::InputTag>("inputJetTags"))
 ,mqq_         (iConfig.getParameter<double>       ("Mqq"         ))
 ,detaqq_      (iConfig.getParameter<double>       ("Detaqq"      ))
 ,detabb_      (iConfig.getParameter<double>       ("Detabb"      ))
 ,ptsqq_       (iConfig.getParameter<double>       ("Ptsumqq"     ))
 ,ptsbb_       (iConfig.getParameter<double>       ("Ptsumbb"     ))
 ,seta_        (iConfig.getParameter<double>       ("Etaq1Etaq2"  ))
 ,value_       (iConfig.getParameter<std::string>  ("value"       ))
 ,triggerType_ (iConfig.getParameter<int>          ("triggerType" ))
{
}


template<typename T>
HLTJetSortedVBFFilter<T>::~HLTJetSortedVBFFilter()
{ }

template<typename T>
void
HLTJetSortedVBFFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(string("hlt")+string(typeid(HLTJetSortedVBFFilter<T>).name()),desc);
}


//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T>
bool 
HLTJetSortedVBFFilter<T>::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs& filterproduct)
{

     using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   typedef vector<T> TCollection;
   typedef Ref<TCollection> TRef;
     
   bool accept(false);

   if (saveTags()) filterproduct.addCollectionTag(inputJets_);

   const unsigned int nMax(4);
   vector<Jpair> sorted(nMax);
   vector<TRef> jetRefs(nMax);

   Handle<TCollection> jets;
   event.getByLabel(inputJets_,jets);
   Handle<JetTagCollection> jetTags;

   unsigned int nJet=0;
   double value(0.0);

   Particle::LorentzVector b1,b2,q1,q2;

   if (inputJetTags_.encode()=="") {
     if (jets->size()<nMax) return false;
     for (typename TCollection::const_iterator jet=jets->begin(); (jet!=jets->end()&& nJet<nMax); ++jet) {
       if (value_=="Pt") {
	 value=jet->pt();
       } else if (value_=="Eta") {
	 value=jet->eta();
       } else if (value_=="Phi") {
	 value=jet->phi();
       } else {
	 value = 0.0;
       }
       sorted[nJet] = make_pair(value,nJet);
       ++nJet;
     }
     sort(sorted.begin(),sorted.end(),comparator);
     for (unsigned int i=0; i<nMax; ++i) {
       jetRefs[i]=TRef(jets,sorted[i].second);
     }
     q1 = jetRefs[3]->p4();
     b1 = jetRefs[2]->p4();
     b2 = jetRefs[1]->p4();
     q2 = jetRefs[0]->p4();
   } else {
     event.getByLabel(inputJetTags_,jetTags);
     if (jetTags->size()<nMax) return false;
     for (JetTagCollection::const_iterator jet = jetTags->begin(); (jet!=jetTags->end()&&nJet<nMax); ++jet) {
       if (value_=="second") {
	 value = jet->second;
       } else {
	 value = 0.0;
       }
       sorted[nJet] = make_pair(value,nJet);
       ++nJet;
     }
     sort(sorted.begin(),sorted.end(),comparator);
     for (unsigned int i=0; i<nMax; ++i) {
       jetRefs[i]= TRef(jets,(*jetTags)[sorted[i].second].first.key());
     }
     b1 = jetRefs[3]->p4();
     b2 = jetRefs[2]->p4();
     q1 = jetRefs[1]->p4();
     q2 = jetRefs[0]->p4();
   }

   double mqq_bs     = (q1+q2).M();
   double deltaetaqq = std::abs(q1.Eta()-q2.Eta());
   double deltaetabb = std::abs(b1.Eta()-b2.Eta());
   double ptsqq_bs   = (q1+q2).Pt();
   double ptsbb_bs   = (b1+b2).Pt();
   double signeta    = q1.Eta()*q2.Eta();
   
   if ( 
	(mqq_bs     > mqq_    ) &&
	(deltaetaqq > detaqq_ ) &&
	(deltaetabb < detabb_ ) &&
	(ptsqq_bs   > ptsqq_  ) &&
	(ptsbb_bs   > ptsbb_  ) &&
	(signeta    < seta_   )
	) {
     accept=true;
     for (unsigned int i=0; i<nMax; ++i) {
       filterproduct.addObject(triggerType_,jetRefs[i]);
     }
   }

   return accept;
}
