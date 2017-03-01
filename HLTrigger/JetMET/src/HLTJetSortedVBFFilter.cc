/** \class HLTJetSortedVBFFilter
 *
 * See header file for documentation
 *


 *
 *  \author Jacopo Bernardini
 *
 */

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "HLTrigger/JetMET/interface/HLTJetSortedVBFFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

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
,dphibb_      (iConfig.getParameter<double>       ("Dphibb"      ))
,ptsqq_       (iConfig.getParameter<double>       ("Ptsumqq"     ))
,ptsbb_       (iConfig.getParameter<double>       ("Ptsumbb"     ))
,seta_        (iConfig.getParameter<double>       ("Etaq1Etaq2"  ))
,njets_       (iConfig.getParameter<int>          ("njets"       ))
,value_       (iConfig.getParameter<std::string>  ("value"       ))
,triggerType_ (iConfig.getParameter<int>          ("triggerType" ))
{
	m_theJetsToken = consumes<std::vector<T>>(inputJets_);
	m_theJetTagsToken = consumes<reco::JetTagCollection>(inputJetTags_);
	if(njets_<4) {
		edm::LogWarning("LowNJets")<< "njets="<<njets_<<" it must be >=4. Forced njets=4.";
		njets_=4;
	}
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
	desc.add<double>("Dphibb",10.);
	desc.add<double>("Ptsumqq",0.);
	desc.add<double>("Ptsumbb",0.);
	desc.add<double>("Etaq1Etaq2",40.);
	desc.add<std::string>("value","second");
	desc.add<int>("triggerType",trigger::TriggerJet);
	desc.add<int>("njets",4);
        descriptions.add(defaultModuleLabel<HLTJetSortedVBFFilter<T>>(), desc);
}

template<typename T> float HLTJetSortedVBFFilter<T>::findCSV(const typename std::vector<T>::const_iterator & jet, const reco::JetTagCollection  & jetTags){
	float minDr = 0.1;
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
HLTJetSortedVBFFilter<T>::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs& filterproduct) const
{
	using namespace std;
	using namespace edm;
	using namespace reco;
	using namespace trigger;

	typedef vector<T> TCollection;
	typedef Ref<TCollection> TRef;

	bool accept(false);
	
	Handle<TCollection> jets;
	event.getByToken(m_theJetsToken,jets);
	Handle<JetTagCollection> jetTags;

	if (saveTags()) filterproduct.addCollectionTag(inputJets_);
	if (jets->size()<4) return false;

	const unsigned int nMax(njets_<jets->size()?njets_:jets->size());
	vector<Jpair> sorted(nMax);
	vector<TRef> jetRefs(nMax);
	unsigned int nJetRefs=0;

	unsigned int nJet=0;
	double value(0.0);

	Particle::LorentzVector b1,b2,q1,q2;
	if (inputJetTags_.encode()=="") {
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
		nJetRefs=nMax;
		q1 = jetRefs[3]->p4();
		b1 = jetRefs[2]->p4();
		b2 = jetRefs[1]->p4();
		q2 = jetRefs[0]->p4();
	} else if(value_=="1BTagAndEta"){
		event.getByToken(m_theJetTagsToken,jetTags);
		vector<Jpair> sorted;
		unsigned int b1_idx=-1;
		float csv_max=-999;
		for (typename TCollection::const_iterator jet=jets->begin(); (jet!=jets->end()&& nJet<nMax); ++jet) { //fill "sorted" and get the most b-tagged jet with higher CSV (b1)
			value = findCSV(jet, *jetTags);
			if(value>csv_max) {
				csv_max=value;
				b1_idx=nJet;
			}
			sorted.push_back(make_pair(jet->eta(),nJet));
			nJet++;
			//   		cout << "jetPt=" << jet->pt() << "\tjetEta=" << jet->eta() << "\tjetCSV=" << value << endl;
		}
		if(b1_idx>=sorted.size() || b1_idx<0) edm::LogError("OutOfRange")<< "b1 index out of range.";
		sorted.erase(sorted.begin()+b1_idx); //remove the most b-tagged jet from "sorted"
		sort(sorted.begin(),sorted.end(),comparator); //sort "sorted" by eta

		unsigned int q1_idx=sorted.front().second;  //take the backward jet (q1)
		unsigned int q2_idx=sorted.back().second;  //take the forward jet (q2)

		unsigned int i=0;
		while( (i==q1_idx) || (i==q2_idx) || (i==b1_idx) ) i++; //take jet with highest pT but q1,q2,b1 (q2)
		unsigned int b2_idx=i;

		if(q1_idx<jets->size()) q1 = jets->at(q1_idx).p4(); else edm::LogWarning("Something wrong with q1");
		if(q2_idx<jets->size()) q2 = jets->at(q2_idx).p4(); else edm::LogWarning("Something wrong with q2");
		if(b1_idx<jets->size()) b1 = jets->at(b1_idx).p4(); else edm::LogWarning("Something wrong with b1");
		if(b2_idx<jets->size()) b2 = jets->at(b2_idx).p4(); else edm::LogWarning("Something wrong with b2");
		
		jetRefs[0]= TRef(jets,b1_idx);
		jetRefs[1]= TRef(jets,b2_idx);
		jetRefs[2]= TRef(jets,q1_idx);
		jetRefs[3]= TRef(jets,q2_idx);
		nJetRefs=4;

		//   	cout<<"\tPathB: b1="<<b1.pt()<<" b2="<<b2.pt()<<" q1="<<q1.pt()<<" q2="<<q2.pt()<<endl; 
	} else if(value_=="2BTagAndPt"){
		event.getByToken(m_theJetTagsToken,jetTags);
		vector<Jpair> sorted;

		unsigned int b1_idx=-1;
		unsigned int b2_idx=-1;
		float csv1=-999;
		float csv2=-999;
		for (typename TCollection::const_iterator jet=jets->begin(); (jet!=jets->end()&& nJet<nMax); ++jet) { //fill "sorted" and get the two most b-tagged jets (b1,b2)
			value = findCSV(jet, *jetTags);
			if(value>csv1) {
				csv2=csv1;
				b2_idx=b1_idx;
				csv1=value;
				b1_idx=nJet;
			} 
			else if(value>csv2){
				csv2=value;
				b2_idx=nJet;
			}
			sorted.push_back(make_pair(jet->eta(),nJet));
			nJet++;
			//   		cout << "jetPt=" << jet->pt() << "\tjetEta=" << jet->eta() << "\tjetCSV=" << value << endl;
		}
		sorted.erase(sorted.begin()+b1_idx); //remove b1 and b2 from sorted
		sorted.erase(sorted.begin()+(b1_idx>b2_idx?b2_idx:b2_idx-1));

		unsigned int q1_idx=sorted.at(0).second;  //get q1 and q2 as the jets with highest pT, but b1 and b2.
		unsigned int q2_idx=sorted.at(1).second;

		if(q1_idx<jets->size()) q1 = jets->at(q1_idx).p4(); else edm::LogWarning("Something wrong with q1");
		if(q2_idx<jets->size()) q2 = jets->at(q2_idx).p4(); else edm::LogWarning("Something wrong with q2");
		if(b1_idx<jets->size()) b1 = jets->at(b1_idx).p4(); else edm::LogWarning("Something wrong with b1");
		if(b2_idx<jets->size()) b2 = jets->at(b2_idx).p4(); else edm::LogWarning("Something wrong with b2");

		jetRefs[0]= TRef(jets,b1_idx);
		jetRefs[1]= TRef(jets,b2_idx);
		jetRefs[2]= TRef(jets,q1_idx);
		jetRefs[3]= TRef(jets,q2_idx);
		nJetRefs=4;

		//   	cout<<"\tPathA: b1="<<b1.pt()<<" b2="<<b2.pt()<<" q1="<<q1.pt()<<" q2="<<q2.pt()<<endl; 
	}
	else {
		event.getByToken(m_theJetTagsToken,jetTags);
		for (typename TCollection::const_iterator jet=jets->begin(); (jet!=jets->end()&& nJet<nMax); ++jet) {

			if (value_=="second") {
				value = findCSV(jet, *jetTags);
			} else {
				value = 0.0;
			}
			sorted[nJet] = make_pair(value,nJet);
			++nJet;
		}
		sort(sorted.begin(),sorted.end(),comparator);
		for (unsigned int i=0; i<nMax; ++i) {
			jetRefs[i]= TRef(jets,sorted[i].second);
		}
		nJetRefs=nMax;
		b1 = jetRefs[3]->p4();
		b2 = jetRefs[2]->p4();
		q1 = jetRefs[1]->p4();
		q2 = jetRefs[0]->p4();
	}

	double mqq_bs     = (q1+q2).M();
	double deltaetaqq = std::abs(q1.Eta()-q2.Eta());
	double deltaetabb = std::abs(b1.Eta()-b2.Eta());
	double deltaphibb = std::abs(reco::deltaPhi(b1.Phi(),b2.Phi()));
	double ptsqq_bs   = (q1+q2).Pt();
	double ptsbb_bs   = (b1+b2).Pt();
	double signeta    = q1.Eta()*q2.Eta();

	if (
			(mqq_bs     > mqq_    ) &&
			(deltaetaqq > detaqq_ ) &&
			(deltaetabb < detabb_ ) &&
			(deltaphibb < dphibb_ ) &&
			(ptsqq_bs   > ptsqq_  ) &&
			(ptsbb_bs   > ptsbb_  ) &&
			(signeta    < seta_   )
	   ) {
		accept=true;
		for (unsigned int i=0; i<nJetRefs; ++i) {
			filterproduct.addObject(triggerType_,jetRefs[i]);
		}
	}

	return accept;
}

