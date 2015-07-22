/** \class HLTJetHbbFilter
 *
 * See header file for documentation
 *
 *  \author Ann Wang
 *
 */

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "HLTrigger/JetMET/plugins/HLTJetHbbFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

using namespace std;
using namespace reco;
using namespace trigger;
//
// constructors and destructor//
//
template<typename T>
HLTJetHbbFilter<T>::HLTJetHbbFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
 ,inputJets_   (iConfig.getParameter<edm::InputTag>("inputJets"   ))
 ,inputJetTags_(iConfig.getParameter<edm::InputTag>("inputJetTags"))
 ,minmbb_      (iConfig.getParameter<double>       ("minMbb"      ))
 ,maxmbb_      (iConfig.getParameter<double>       ("maxMbb"      ))
 ,minptb1_     (iConfig.getParameter<double>       ("minPtb1"     ))
 ,minptb2_     (iConfig.getParameter<double>       ("minPtb2"     ))
 ,maxetab_     (iConfig.getParameter<double>       ("maxEtab"     ))
 ,minptbb_     (iConfig.getParameter<double>       ("minPtbb"     ))
 ,maxptbb_     (iConfig.getParameter<double>       ("maxPtbb"     ))
 ,mintag1_     (iConfig.getParameter<double>       ("minTag1"     ))
 ,mintag2_     (iConfig.getParameter<double>       ("minTag2"     ))
 ,maxtag_      (iConfig.getParameter<double>       ("maxTag"      ))
 ,triggerType_ (iConfig.getParameter<int>          ("triggerType" ))
{
  m_theJetsToken = consumes<std::vector<T>>(inputJets_);
  m_theJetTagsToken = consumes<reco::JetTagCollection>(inputJetTags_);
  
  //put a dummy METCollection into the event, holding values for csv tag 1 and tag 2 values
  produces<reco::METCollection>();
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
  desc.add<double>("minMbb",70);
  desc.add<double>("maxMbb",200);
  desc.add<double>("minPtb1",-1);
  desc.add<double>("minPtb2",-1);
  desc.add<double>("maxEtab",99999.0);
  desc.add<double>("minPtbb",-1);
  desc.add<double>("maxPtbb",-1);
  desc.add<double>("minTag1",0.5);
  desc.add<double>("minTag2",0.2);
  desc.add<double>("maxTag",99999.0);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTJetHbbFilter<T>>(), desc);
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
  double ptjet1  = -99.;
  double etajet1 = -99.;

  double ejet2   = -99.;
  double pxjet2  = -99.;
  double pyjet2  = -99.;
  double pzjet2  = -99.;
  double ptjet2  = -99.;
  double etajet2 = -99.;

  //looping through sets of jets
  for (typename TCollection::const_iterator jet1=jets->begin(); (jet1!=jets->end()); ++jet1) {
    tag1 = findCSV(jet1, *jetTags);
    ++nJet;
    for (typename TCollection::const_iterator jet2=(jet1+1); (jet2!=jets->end()); ++jet2) {
      tag2 = findCSV(jet2, *jetTags);

      ejet1   = jet1->energy();
      pxjet1  = jet1->px();
      pyjet1  = jet1->py();
      pzjet1  = jet1->pz();
      ptjet1  = jet1->pt();
      etajet1  = jet1->eta();

      ejet2   = jet2->energy();
      pxjet2  = jet2->px();
      pyjet2  = jet2->py();
      pzjet2  = jet2->pz();
      ptjet2  = jet2->pt();
      etajet2  = jet2->eta();


      if ( ( (mintag1_ <= tag1) and (tag1 <= maxtag_) ) && ( (mintag2_ <= tag2) and (tag2 <= maxtag_) ) ) {// if they're both b's
	if ( fabs(etajet1) <= maxetab_ && fabs(etajet2) <= maxetab_ ) { // if they satisfy the eta requirement
	  if ( ( ptjet1 >= minptb1_ && ptjet2 >= minptb2_ ) || ( ptjet2 >= minptb1_ && ptjet1 >= minptb2_ ) ) { // if they satisfy the pt requirement
	  
	    double ptbb = sqrt( (pxjet1 + pxjet2) * (pxjet1 + pxjet2) +
				(pyjet1 + pyjet2) * (pyjet1 + pyjet2) ); // pt of the two jets

	    if ( ptbb >= minptbb_ && ( maxptbb_ < 0 || ptbb <= maxptbb_ ) ) { //if they satisfy the vector pt requirement
       
	      double mbb = sqrt( (ejet1  + ejet2)  * (ejet1  + ejet2) -
				 (pxjet1 + pxjet2) * (pxjet1 + pxjet2) -
				 (pyjet1 + pyjet2) * (pyjet1 + pyjet2) - 
				 (pzjet1 + pzjet2) * (pzjet1 + pzjet2) );// mass of two jets
                                                  
	      if ( (minmbb_ <= mbb) and (mbb <= maxmbb_ ) ) { // if they fit the mass requirement          
		accept = true;

		TRef ref1 = TRef(jets, distance(jets->begin(),jet1));
		TRef ref2 = TRef(jets, distance(jets->begin(),jet2));
	      
		if (saveTags()) filterproduct.addCollectionTag(inputJets_);
		filterproduct.addObject(triggerType_,ref1);
		filterproduct.addObject(triggerType_,ref2);
	      
		//create METCollection for storing csv tag1 and tag2 results
		std::auto_ptr<reco::METCollection> csvObject(new reco::METCollection());
		reco::MET::LorentzVector csvP4(tag1,tag2,0,0);
		reco::MET::Point vtx(0,0,0);
		reco::MET csvTags(csvP4, vtx);
		csvObject->push_back(csvTags);
		edm::RefProd<reco::METCollection > ref_before_put = event.getRefBeforePut<reco::METCollection >();
		//put the METCollection into the event (necessary because of how addCollectionTag works...)
		event.put(csvObject);
		edm::Ref<reco::METCollection> csvRef(ref_before_put, 0);
		if (saveTags()) filterproduct.addCollectionTag(edm::InputTag( *moduleLabel()));
		filterproduct.addObject(trigger::TriggerMET, csvRef); //give it the ID of a MET object
		return accept;
	      }
	    }
	  }
	}
      }
    }
  }
  return accept;
}
typedef HLTJetHbbFilter<CaloJet> HLTCaloJetHbbFilter;
typedef HLTJetHbbFilter<  PFJet> HLTPFJetHbbFilter;

DEFINE_FWK_MODULE(HLTCaloJetHbbFilter);
DEFINE_FWK_MODULE(HLTPFJetHbbFilter);
