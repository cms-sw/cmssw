/** \class HLTJetHbbFilter
 *
 * See header file for documentation
 *
 *  \author Ann Wang
 *
 */


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "HLTrigger/JetMET/interface/HLTJetHbbFilter.h"

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
HLTJetHbbFilter<T>::HLTJetHbbFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
 ,inputJets_   (iConfig.getParameter<edm::InputTag>("inputJets"   ))
 ,inputJetTags_(iConfig.getParameter<edm::InputTag>("inputJetTags"))
 ,minmqq_      (iConfig.getParameter<double>       ("minMqq"      ))
 ,maxmqq_      (iConfig.getParameter<double>       ("maxMqq"      ))
 ,detaqq_      (iConfig.getParameter<double>       ("Detaqq"      ))
 ,detabb_      (iConfig.getParameter<double>       ("Detabb"      ))
 ,dphibb_      (iConfig.getParameter<double>       ("Dphibb"      )) 	
 ,ptsqq_       (iConfig.getParameter<double>       ("Ptsumqq"     ))
 ,ptsbb_       (iConfig.getParameter<double>       ("Ptsumbb"     ))
 ,seta_        (iConfig.getParameter<double>       ("Etaq1Etaq2"  ))
 ,value_       (iConfig.getParameter<std::string>  ("value"       ))
 ,mintag1_     (iConfig.getParameter<double>       ("minTag1"     ))
 ,mintag2_     (iConfig.getParameter<double>       ("minTag2"     ))
 ,maxtag_      (iConfig.getParameter<double>       ("maxTag"      ))
 ,triggerType_ (iConfig.getParameter<int>          ("triggerType" ))
{
  m_theJetsToken = consumes<std::vector<T>>(inputJets_);
  m_theJetTagsToken = consumes<reco::JetTagCollection>(inputJetTags_);
}


template<typename T>
HLTJetHbbFilter<T>::~HLTJetHbbFilter()
{ }

template<typename T>
void
HLTJetHbbFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJets",edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("inputJetTags",edm::InputTag(""));
  desc.add<double>("minMqq",200);
  desc.add<double>("maxMqq",200);
  desc.add<double>("Detaqq",2.5);
  desc.add<double>("Detabb",10.);
  desc.add<double>("Dphibb",10.);
  desc.add<double>("Ptsumqq",0.);
  desc.add<double>("Ptsumbb",0.);
  desc.add<double>("Etaq1Etaq2",40.);
  desc.add<std::string>("value","second");
  desc.add<double>("minTag1",0.7);
  desc.add<double>("minTag2",0.4);
  desc.add<double>("maxTag",9999.0);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(string("hlt")+string(typeid(HLTJetHbbFilter<T>).name()),desc);
}

template<typename T> float HLTJetHbbFilter<T>::findCSV(const typename std::vector<T>::const_iterator & jet, const reco::JetTagCollection  & jetTags){
        float minDr = 0.1; //matching jet tag with jet
        float tmpCSV = -20 ;
        for (reco::JetTagCollection::const_iterator jetb = jetTags.begin(); (jetb!=jetTags.end()); ++jetb) {
        float tmpDr = reco::deltaR(*jet,*(jetb->first));
        if (tmpDr < minDr) {
                minDr = tmpDr ;
                tmpCSV= jetb->second;
                }
        }
        return tmpCSV;
}
//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTJetHbbFilter<T>::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs& filterproduct) const
{

   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   typedef vector<T> TCollection;
   typedef Ref<TCollection> TRef;

   bool accept(false);
   //const unsigned int nMax(15);

   if (saveTags()) filterproduct.addCollectionTag(inputJets_);

   Handle<TCollection> jets;
   event.getByToken(m_theJetsToken,jets);
   Handle<JetTagCollection> jetTags;

   unsigned int nJet=0;

   event.getByToken(m_theJetTagsToken,jetTags);

   double tag1    = -99.;
   double tag2    = -99.;

   if (jetTags->size()<2) return false;

   double ejet1   = -99.;
   double pxjet1  = -99.;
   double pyjet1  = -99.;
   double pzjet1  = -99.;

   double ejet2   = -99.;
   double pxjet2  = -99.;
   double pyjet2  = -99.;
   double pzjet2  = -99.;

   //looping through sets of jets
   for (typename TCollection::const_iterator jet1=jets->begin(); (jet1!=jets->end()); ++jet1) {
     if (value_=="second") {
       tag1 = findCSV(jet1, *jetTags);
     }
     ++nJet;
     for (typename TCollection::const_iterator jet2=(jet1+1); (jet2!=jets->end()); ++jet2) {
       tag2 = findCSV(jet2, *jetTags);

       ejet1   = jet1->energy();
       pxjet1  = jet1->px();
       pyjet1  = jet1->py();
       pzjet1  = jet1->pz();

       ejet2   = jet2->energy();
       pxjet2  = jet2->px();
       pyjet2  = jet2->py();
       pzjet2  = jet2->pz();


       double mbb = sqrt( (ejet1  + ejet2)  * (ejet1  + ejet2) -
			  (pxjet1 + pxjet2) * (pxjet1 + pxjet2) -
			  (pyjet1 + pyjet2) * (pyjet1 + pyjet2) - 
			  (pzjet1 + pzjet2) * (pzjet1 + pzjet2) );// mass of two jets
                                                  
       if ( ( (mintag1_ <= tag1) and (tag1 <= maxtag_) ) &&
	    ( (mintag2_ <= tag2) and (tag2 <= maxtag_) ) &&
	    ( (minmqq_ <= mbb) and (mbb <= maxmqq_ ) ) ) { // if they're both bs and they fit the mass requirement          
            
	 accept = true;	 
	 TRef ref1 = TRef(jets, distance(jets->begin(),jet1));
	 TRef ref2 = TRef(jets, distance(jets->begin(),jet2));
	 filterproduct.addObject(triggerType_,ref1);                                                                        
	 filterproduct.addObject(triggerType_,ref2);  
       }

     }
   }

   return accept;
}
