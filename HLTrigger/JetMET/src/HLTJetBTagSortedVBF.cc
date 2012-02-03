/** \class HLTJetBTagSortedVBF
 *
 * See header file for documentation
 *
 *  $Date: 2012/02/03 10:38:50 $
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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HLTrigger/JetMET/interface/HLTJetBTagSortedVBF.h"
#include <vector>


using namespace std;

 typedef std::pair<double,int> Jpair;
bool comparatorVBF ( const Jpair& l, const Jpair& r) 
   { return l.first < r.first; }   
//
// constructors and destructor//
//
HLTJetBTagSortedVBF::HLTJetBTagSortedVBF(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
 {
   inputTag_  =(iConfig.getParameter<edm::InputTag>("inputTag"));
   mqq =(iConfig.getParameter<double>       ("Mqq"   ));
   detaqq   = (iConfig.getParameter<double>       ("Detaqq"   ));
   ptsqq    =(iConfig.getParameter<double>       ("Ptsumqq"   ));
   ptsbb    =(iConfig.getParameter<double>       ("Ptsumbb"   ));
   seta    =(iConfig.getParameter<double>       ("Etaq1Etaq2"   ));
}



HLTJetBTagSortedVBF::~HLTJetBTagSortedVBF()
{
}


void
HLTJetBTagSortedVBF::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltBLifetimeL3BJetTagsHbbVBF"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("Mqq",200);
  desc.add<double>("Detaqq",2.5);
  desc.add<double>("Ptsumqq",0.);
  desc.add<double>("Ptsumbb",0.);
  desc.add<double>("Etaq1Etaq2",40.);
  descriptions.add("hltBTagSortedVBF",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTJetBTagSortedVBF::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs & filterproduct)
{

   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   if (saveTags()) filterproduct.addCollectionTag(inputTag_);
   
   bool accept(false);
   vector<Jpair> BTaggedSorted;
   BTaggedSorted.clear();

   // Ref to Candidate object to be recorded in filter object
   edm::Handle<JetTagCollection> h_jetTag;
   event.getByLabel(inputTag_, h_jetTag);
   const reco::JetTagCollection & jetTags = * h_jetTag;
   
   // filter decision

   int nBJet = 0;
   if (jetTags.size()>1) {
     for (JetTagCollection::const_iterator jet = jetTags.begin(); (jet != jetTags.end() && nBJet<4); ++jet) {
       ++nBJet;
       BTaggedSorted.push_back(make_pair(jet->second,nBJet-1));
     }   
   }


     sort(BTaggedSorted.begin(), BTaggedSorted.end(),comparatorVBF);
     Particle::LorentzVector b1,b2,q1,q2;
     int nb1=BTaggedSorted[3].second;int nb2=BTaggedSorted[2].second;int nq1=BTaggedSorted[1].second;int nq2=BTaggedSorted[0].second;
     
     const JetTag jb1=jetTags[nb1];const JetTag jb2=jetTags[nb2];const JetTag jq1=jetTags[nq1];const JetTag jq2=jetTags[nq2];
     b1 = Particle::LorentzVector(jb1.first->px(),jb1.first->py(),jb1.first->pz(),jb1.first->energy());
     b2 = Particle::LorentzVector(jb2.first->px(),jb2.first->py(),jb2.first->pz(),jb2.first->energy());
     q1 = Particle::LorentzVector(jq1.first->px(),jq1.first->py(),jq1.first->pz(),jq1.first->energy());
     q2 = Particle::LorentzVector(jq2.first->px(),jq2.first->py(),jq2.first->pz(),jq2.first->energy());
     
     double ptsbb_bs= (b1+b2).Pt();
     double deltaetaqq=fabs(q1.Eta()-q2.Eta());
     double ptsqq_bs= (q1+q2).Pt();
     double mqq_bs= (q1+q2).M();
     double signeta=q1.Eta()*q2.Eta();
     
     if ((deltaetaqq>detaqq)&&(ptsqq_bs>ptsqq)&&(ptsbb_bs>ptsbb)&&(mqq_bs>mqq)&&(signeta<seta)){
       nBJet=0;
       for (JetTagCollection::const_iterator jet = jetTags.begin(); (jet != jetTags.end() && nBJet<4); ++jet) {
	 ++nBJet;
	 filterproduct.addObject(TriggerJet,jet->first.castTo<reco::CaloJetRef>() );
       }
       accept=true;
     }

       
     // put filter object into the Event
     
     return accept;
}



