/** \class HLTJetEtaSortedVBF
 *
 * See header file for documentation
 *
 *  $Date: 2012/02/03 10:38:26 $


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
#include "HLTrigger/JetMET/interface/HLTJetEtaSortedVBF.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>


using namespace std;

 typedef std::pair<double,int> Jpair;
bool comparator ( const Jpair& l, const Jpair& r) 
   { return l.first < r.first; }   
//
// constructors and destructor//
//
HLTJetEtaSortedVBF::HLTJetEtaSortedVBF(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  inputTag_  =(iConfig.getParameter<edm::InputTag>("inputTag"));
  mqq =(iConfig.getParameter<double>       ("Mqq"   ));
  detaqq    =(iConfig.getParameter<double>       ("Detaqq"   ));
  detabb    =(iConfig.getParameter<double>       ("Detabb"   ));
  ptsqq    =(iConfig.getParameter<double>       ("Ptsumqq"   ));
  ptsbb    =(iConfig.getParameter<double>       ("Ptsumbb"   ));
  seta    =(iConfig.getParameter<double>       ("Etaq1Etaq2"   ));

  
}

HLTJetEtaSortedVBF::~HLTJetEtaSortedVBF()
{
}

void
HLTJetEtaSortedVBF::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltCaloJetL1FastJetCorrected"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("Mqq",200);
  desc.add<double>("Detaqq",2.5);
  desc.add<double>("Detabb",10.);
  desc.add<double>("Ptsumqq",0.);
  desc.add<double>("Ptsumbb",0.);
  desc.add<double>("Etaq1Etaq2",40.);
  descriptions.add("hltEtaSortedVBF",desc);
}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTJetEtaSortedVBF::hltFilter(edm::Event& event, const edm::EventSetup& setup,trigger::TriggerFilterObjectWithRefs& filterproduct)
{

   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;
   if (saveTags()) filterproduct.addCollectionTag(inputTag_);

   bool accept(false);
   vector<Jpair> EtaSorted;
   EtaSorted.clear();

   
   Handle<CaloJetCollection> jets;
   event.getByLabel (inputTag_,jets);
   int nJetCalo=0;
   const reco::CaloJetCollection & jetsc = * jets;
   CaloJetCollection::const_iterator i ( jets->begin() );
   for (; (i!=jets->end() && nJetCalo<4); i++) {
       ++nJetCalo;
       EtaSorted.push_back( make_pair(i->eta(), nJetCalo-1));
   }
   
   sort(EtaSorted.begin(),EtaSorted.end(),comparator);
   Particle::LorentzVector b1,b2,q1,q2;
   int nq1=EtaSorted[3].second;int nb1=EtaSorted[2].second;int nb2=EtaSorted[1].second;int nq2=EtaSorted[0].second;
   const CaloJet jb1=jetsc[nb1];const CaloJet jb2=jetsc[nb2];const CaloJet jq1=jetsc[nq1];const CaloJet jq2=jetsc[nq2];;
   
   b1 = Particle::LorentzVector(jb1.px(),jb1.py(),jb1.pz(),jb1.energy());
   b2 = Particle::LorentzVector(jb2.px(),jb2.py(),jb2.pz(),jb2.energy());
   q1 = Particle::LorentzVector(jq1.px(),jq1.py(),jq1.pz(),jq1.energy());
   q2 = Particle::LorentzVector(jq2.px(),jq2.py(),jq2.pz(),jq2.energy());
   
   double ptsbb_bs= (b1+b2).Pt();
   double deltaetaqq=fabs(q1.Eta()-q2.Eta());
   double deltaetabb=fabs(b1.Eta()-b2.Eta());
   double ptsqq_bs= (q1+q2).Pt();
   double mqq_bs= (q1+q2).M();
   double signeta=q1.Eta()*q2.Eta();
   if ((deltaetaqq>detaqq)&&(ptsqq_bs>ptsqq)&&(ptsbb_bs>ptsbb)&&(mqq_bs>mqq)&&(deltaetabb<detabb)&&(signeta<seta)){
     CaloJetCollection::const_iterator i ( jets->begin() );
     for (; (i!=jets->end() && nJetCalo<4); i++) {
       ++nJetCalo;
       reco::CaloJetRef ref(reco::CaloJetRef(jets,distance(jets->begin(),i)));
       filterproduct.addObject(TriggerJet,ref);
     }
    
     accept=true;
     
   }


   return accept;
}



