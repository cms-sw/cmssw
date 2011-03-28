#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include <vector>

using namespace edm;
using namespace std;
using namespace reco;


class EventNjetAndMetInfoNtupleDumper : public edm::EDProducer {
public:
  EventNjetAndMetInfoNtupleDumper( const edm::ParameterSet & );
   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  edm::InputTag /*muonTag_,*/ metTag_, jetTag_;
  double eJetMin_;

};

EventNjetAndMetInfoNtupleDumper::EventNjetAndMetInfoNtupleDumper( const ParameterSet & cfg ) : 
  //  muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag")),
  metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag")), 
  jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag")),
  eJetMin_(cfg.getUntrackedParameter<double>("EJetMin"))
{
  produces<int>( "numJets" ).setBranchAlias( "numJets" );
  produces<float>( "metEt" ).setBranchAlias( "metEt" );
  produces<float>( "metPhi" ).setBranchAlias( "metPhi" );
  produces<vector<float> >( "jetsBtag" ).setBranchAlias( "jetsBtag" );
  produces<vector <float> >( "jetsEt" ).setBranchAlias( "jetsPt" );
  produces<vector <float> >( "jetsEta" ).setBranchAlias( "jetsEta" );
  produces<vector <float> >( "jetsPhi" ).setBranchAlias( "jetsPhi" );
 
}



void EventNjetAndMetInfoNtupleDumper::produce( Event & evt, const EventSetup & ) {
  auto_ptr<int> nJets( new int );
  auto_ptr<vector<float> > jetsbtag( new vector<float> );
  auto_ptr<vector <float> > jetset( new vector<float> );
  auto_ptr<vector <float> >  jetseta( new vector<float> );
  auto_ptr<vector <float> >  jetsphi( new vector<float> );
  auto_ptr<float> metet( new float );  
  auto_ptr<float> metphi( new float );  

  // MET

  Handle<View<MET> > metCollection;
  if (!evt.getByLabel(metTag_, metCollection)) {
    // LogWarning("") << ">>> MET collection does not exist !!!";
    return;
  }
  const MET& met = metCollection->at(0);
  *metet = -1;  
  *metet = met.et();
  *metphi = met.phi();
 


 // Muon collection, needed for pfjet correct counting
//  Handle<View<Muon> > muonCollection;
//  if (!evt.getByLabel(muonTag_, muonCollection)) {
//    //LogWarning("") << ">>> Muon collection does not exist !!!";
//    return;
//  }
//  unsigned int muonCollectionSize = muonCollection->size();

//  // Get b tag information
//  edm::Handle<reco::JetTagCollection> bTagHandle;
// evt.getByLabel("combinedSecondaryVertexBJetTags", bTagHandle);
// // evt.getByLabel("trackCountingHighEffBJetTags", bTagHandle);
//  const reco::JetTagCollection & bTags = *(bTagHandle.product());

 Handle<View<pat::Jet> >  jetCollection;
 if (!evt.getByLabel(jetTag_, jetCollection)) {
   std::cout << ">>> JET collection does not exist !!!" << std::endl;
   return;
 }
 unsigned int jetCollectionSize = jetCollection->size();
 int njets = 0;
 for (unsigned int i=0; i<jetCollectionSize; i++) {
   const pat::Jet& jet = jetCollection->at(i);
   //  edm::RefToBase<pat::Jet> jetRef = jetCollection->refAt(i);
   // double minDistance=99999; // This is in order to use PFJets
//    for (unsigned int i=0; i<muonCollectionSize; i++) {
//      const Muon& mu = muonCollection->at(i);
//      double distance = sqrt( (mu.eta()-jet.eta())*(mu.eta()-jet.eta()) +(mu.phi()-jet.phi())*(mu.phi()-jet.phi()) );      
//      if (minDistance>distance) minDistance=distance;
//    }
//    if (minDistance<0.3) continue; // 0.3 is the isolation cone around the muon
   if (jet.et()>eJetMin_){

 njets++;
   jetset->push_back(jet.et());
   jetseta->push_back(jet.eta());
   jetsphi->push_back(jet.phi());
   jetsbtag->push_back(jet.bDiscriminator("combinedSecondaryVertexBJetTags"));


   }
 }
 
 *nJets = -1;  
 *nJets = njets;




 // // Loop over jets and study b tag info.
//  for (unsigned int i = 0; i != bTags.size(); ++i) {
//    // cout<<" Jet "<< i 
//    //  <<" has b tag discriminator (trackCountingHighEffBJetTags)= "<<bTags[i].second
//    //  << " and jet Pt = "<<bTags[i].first->pt()<<endl;
  
//    if ( bTags[i].first->et() > eJetMin_ ){
    
//    jetsbtag->push_back(*bTags[i].second);


//    }
//  }



 evt.put( metet, "metEt" );
 evt.put( metphi, "metPhi" );
 evt.put( nJets, "numJets" );
 evt.put( jetsbtag, "jetsBtag" );
 evt.put( jetset, "jetsEt" );
 evt.put( jetseta, "jetsEta" );
 evt.put( jetsphi, "jetsPhi" );

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( EventNjetAndMetInfoNtupleDumper );

