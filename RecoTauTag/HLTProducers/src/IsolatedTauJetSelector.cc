#include "RecoTauTag/HLTProducers/interface/IsolatedTauJetsSelector.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
//
// class decleration
//
using namespace reco;
using namespace edm;


IsolatedTauJetsSelector::IsolatedTauJetsSelector(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  matching_cone      = iConfig.getParameter<double>("MatchingCone");
  signal_cone        = iConfig.getParameter<double>("SignalCone");
  isolation_cone     = iConfig.getParameter<double>("IsolationCone"); 
  pt_min_isolation   = iConfig.getParameter<double>("MinimumTransverseMomentumInIsolationRing"); 
  pt_min_leadTrack   = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack"); 
  n_tracks_isolation_ring = iConfig.getParameter<int>("MaximumNumberOfTracksIsolationRing"); 
  dZ_vertex          = iConfig.getParameter<double>("DeltaZetTrackVertex");//To be modified
 
  produces<reco::CaloJetCollection>();
  produces<reco::JetTagCollection>(); 
  
}

IsolatedTauJetsSelector::~IsolatedTauJetsSelector(){ }

void IsolatedTauJetsSelector::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;

 auto_ptr<CaloJetCollection> selectedTaus(new CaloJetCollection);
 std::auto_ptr<reco::JetTagCollection> resultBase(new JetTagCollection);

 for( vtag::const_iterator s = jetSrc.begin(); s != jetSrc.end(); ++ s ) {
   edm::Handle<IsolatedTauTagInfoCollection> tauJets;
   iEvent.getByLabel( * s, tauJets );
   IsolatedTauTagInfoCollection::const_iterator i = tauJets->begin();
   for(;i !=tauJets->end(); i++ ) {

     JetTracksAssociationRef     jetTracks = i->jetRef()->jtaRef();
     math::XYZVector jetDir(jetTracks->key->px(),jetTracks->key->py(),jetTracks->key->pz());   
     float discriminator = i->discriminator(jetDir, matching_cone, signal_cone, isolation_cone, pt_min_leadTrack, pt_min_isolation,  n_tracks_isolation_ring,dZ_vertex); 
     if(discriminator > 0) {
       JetTag pippoTag(discriminator,jetTracks);
       resultBase->push_back(pippoTag);
       const CaloJet* pippo = dynamic_cast<const CaloJet*>(&(i->jet()));
       selectedTaus->push_back(*pippo );
     }
     
   }
 } 


  iEvent.put(selectedTaus);
  iEvent.put(resultBase);
}
