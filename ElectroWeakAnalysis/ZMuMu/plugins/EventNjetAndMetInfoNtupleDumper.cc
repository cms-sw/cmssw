#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"


#include <vector>

using namespace edm;
using namespace std;
using namespace reco;


class EventNjetAndMetInfoNtupleDumper : public edm::EDProducer {
public:
  EventNjetAndMetInfoNtupleDumper( const edm::ParameterSet & );
   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  edm::InputTag muonTag_, metTag_, jetTag_;
  double eJetMin_;

};

EventNjetAndMetInfoNtupleDumper::EventNjetAndMetInfoNtupleDumper( const ParameterSet & cfg ) : 
  muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag")),
  metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag")), 
  jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag")),
  eJetMin_(cfg.getUntrackedParameter<double>("EJetMin"))
{
  produces<int>( "numJets" ).setBranchAlias( "numJets" );
  produces<float>( "metPt" ).setBranchAlias( "metPt" );
 
}



void EventNjetAndMetInfoNtupleDumper::produce( Event & evt, const EventSetup & ) {
  auto_ptr<int> nJets( new int );
  auto_ptr<float> metpt( new float );  

  // MET

  Handle<View<MET> > metCollection;
  if (!evt.getByLabel(metTag_, metCollection)) {
    // LogWarning("") << ">>> MET collection does not exist !!!";
    return;
  }
  const MET& met = metCollection->at(0);
  *metpt = -1;  
  *metpt = met.pt();
 


 // Muon collection, needed for pfjet correct counting
 Handle<View<Muon> > muonCollection;
 if (!evt.getByLabel(muonTag_, muonCollection)) {
   //LogWarning("") << ">>> Muon collection does not exist !!!";
   return;
 }
 unsigned int muonCollectionSize = muonCollection->size();



 Handle<View<Jet> > jetCollection;
 if (!evt.getByLabel(jetTag_, jetCollection)) {
   //LogError("") << ">>> JET collection does not exist !!!";
   return;
 }
 unsigned int jetCollectionSize = jetCollection->size();
 int njets = 0;
 for (unsigned int i=0; i<jetCollectionSize; i++) {
   const Jet& jet = jetCollection->at(i);
   double minDistance=99999; // This is in order to use PFJets
   for (unsigned int i=0; i<muonCollectionSize; i++) {
     const Muon& mu = muonCollection->at(i);
     double distance = sqrt( (mu.eta()-jet.eta())*(mu.eta()-jet.eta()) +(mu.phi()-jet.phi())*(mu.phi()-jet.phi()) );      
     if (minDistance>distance) minDistance=distance;
   }
   if (minDistance<0.3) continue; // 0.3 is the isolation cone around the muon
   if (jet.et()>eJetMin_) njets++;
 }
 
 *nJets = -1;  
 *nJets = njets;

 evt.put( metpt, "metPt" );
 evt.put( nJets, "numJets" );

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( EventNjetAndMetInfoNtupleDumper );

