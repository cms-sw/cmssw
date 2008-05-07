#include "RecoTauTag/HLTProducers/interface/HLTTauProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//


HLTTauProducer::HLTTauProducer(const edm::ParameterSet& iConfig)
{
  emIsolatedJetsL2_ = iConfig.getParameter<edm::InputTag>("L2EcalIsoJets");
  trackIsolatedJetsL25_ = iConfig.getParameter<edm::InputTag>("L25TrackIsoJets");
  trackIsolatedJetsL3_ = iConfig.getParameter<edm::InputTag>("L3TrackIsoJets");
  matchingCone_ = iConfig.getParameter<double>("MatchingCone");
  signalCone_ = iConfig.getParameter<double>("SignalCone");
  isolationCone_ = iConfig.getParameter<double>("IsolationCone");
  produces<reco::HLTTauCollection>();
}

HLTTauProducer::~HLTTauProducer(){ }

void HLTTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  

  HLTTauCollection * jetCollection = new HLTTauCollection;
 
  edm::Handle<L2TauInfoAssociation> tauL2Jets;
  iEvent.getByLabel(emIsolatedJetsL2_ , tauL2Jets );

  edm::Handle<IsolatedTauTagInfoCollection> tauL25Jets;
  iEvent.getByLabel(trackIsolatedJetsL25_, tauL25Jets );
  
  edm::Handle<IsolatedTauTagInfoCollection> tauL3Jets;
  iEvent.getByLabel(trackIsolatedJetsL3_, tauL3Jets );

  IsolatedTauTagInfoCollection tauL25 = *(tauL25Jets.product());
  IsolatedTauTagInfoCollection tauL3 = *(tauL3Jets.product());
  
  int i=0;
  float eta_, phi_, pt_;

 for(L2TauInfoAssociation::const_iterator p = tauL2Jets->begin();p!=tauL2Jets->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     //	     const CaloJet& jet =*(p->key);
	     

	     double emIsol  = l2info.ECALIsolConeCut;

    JetTracksAssociationRef jetTracks = tauL25[i].jtaRef();
    math::XYZVector jetDirL25(jetTracks->first->px(),jetTracks->first->py(),jetTracks->first->pz());   
    eta_ = jetDirL25.eta();
    phi_ = jetDirL25.phi();
    pt_ = jetTracks->first->pt();

    int trackIsolationL25 = (int)tauL25[i].discriminator(jetDirL25,matchingCone_, signalCone_, isolationCone_,1.,1.,0);
    const TrackRef leadTkL25 = tauL25[i].leadingSignalTrack(jetDirL25,matchingCone_, 1.);
    double ptLeadTkL25=0.;
    
    if(!leadTkL25) 
      {}else{
	ptLeadTkL25 = (*leadTkL25).pt();
      }
    jetTracks = tauL3[i].jtaRef();
    math::XYZVector jetDirL3(jetTracks->first->px(),jetTracks->first->py(),jetTracks->first->pz());	
    int  trackIsolationL3 = (int)tauL3[i].discriminator(jetDirL3,matchingCone_, signalCone_, isolationCone_,ptMinLeadTk_,1.,0);
    const TrackRef leadTkL3 = tauL3[i].leadingSignalTrack(jetDirL3,matchingCone_, ptMinLeadTk_);
    double ptLeadTkL3=0.;
    if(!leadTkL3) 
      {}else{
	ptLeadTkL3 = (*leadTkL3).pt();
      }

    HLTTau pippo(eta_,phi_,pt_,emIsol,trackIsolationL25,ptLeadTkL25,trackIsolationL3,ptLeadTkL3);
      jetCollection->push_back(pippo);
      i++;
  }
  
  auto_ptr<reco::HLTTauCollection> selectedTaus(jetCollection);
  
  iEvent.put(selectedTaus);
  



}
