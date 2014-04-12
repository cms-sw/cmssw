#include "RecoTauTag/HLTProducers/interface/HLTTauProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//


HLTTauProducer::HLTTauProducer(const edm::ParameterSet& iConfig)
{
  emIsolatedJetsL2_ = consumes<reco::L2TauInfoAssociation>(iConfig.getParameter<edm::InputTag>("L2EcalIsoJets") );
  trackIsolatedJetsL25_ = consumes<reco::IsolatedTauTagInfoCollection>(iConfig.getParameter<edm::InputTag>("L25TrackIsoJets") );
  trackIsolatedJetsL3_ = consumes<reco::IsolatedTauTagInfoCollection>(iConfig.getParameter<edm::InputTag>("L3TrackIsoJets") );
  matchingCone_ = iConfig.getParameter<double>("MatchingCone");
  signalCone_ = iConfig.getParameter<double>("SignalCone");
  isolationCone_ = iConfig.getParameter<double>("IsolationCone");
  ptMin_ = iConfig.getParameter<double>("MinPtTracks");
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
  iEvent.getByToken(emIsolatedJetsL2_ , tauL2Jets );

  edm::Handle<IsolatedTauTagInfoCollection> tauL25Jets;
  iEvent.getByToken(trackIsolatedJetsL25_, tauL25Jets );
  
  edm::Handle<IsolatedTauTagInfoCollection> tauL3Jets;
  iEvent.getByToken(trackIsolatedJetsL3_, tauL3Jets );

  IsolatedTauTagInfoCollection tauL25 = *(tauL25Jets.product());
  IsolatedTauTagInfoCollection tauL3 = *(tauL3Jets.product());
  
  int i=0;
  float eta_, phi_, pt_;
  int nTracksL25, nTracksL3;
    float sumPtTracksL25 = 1000.;
    float sumPtTracksL3 = 1000.;
    double ptLeadTkL25=0.;
    double ptLeadTkL3=0.;
 for(L2TauInfoAssociation::const_iterator p = tauL2Jets->begin();p!=tauL2Jets->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     //	     const CaloJet& jet =*(p->key);
	     

	     double emIsol  = l2info.ecalIsolEt();

    JetTracksAssociationRef jetTracks = tauL25[i].jtaRef();
    math::XYZVector jetDirL25(jetTracks->first->px(),jetTracks->first->py(),jetTracks->first->pz());   
    eta_ = jetDirL25.eta();
    phi_ = jetDirL25.phi();
    pt_ = jetTracks->first->pt();

    int trackIsolationL25 = (int)tauL25[i].discriminator(jetDirL25,matchingCone_, signalCone_, isolationCone_,1.,ptMin_,0);
    const TrackRef leadTkL25 = tauL25[i].leadingSignalTrack(jetDirL25,matchingCone_, 1.);
    ptLeadTkL25 = 0.;
    nTracksL25 = 1000;
    if(!leadTkL25) 
      {}else{
	ptLeadTkL25 = (*leadTkL25).pt();
	const TrackRefVector signalTracks = tauL25[i].tracksInCone((*leadTkL25).momentum(), signalCone_, ptMin_ );
	const TrackRefVector isolationTracks = tauL25[i].tracksInCone((*leadTkL25).momentum(), isolationCone_,   ptMin_ );
	nTracksL25 = isolationTracks.size() - signalTracks.size();

	for(unsigned int j=0;j<isolationTracks.size();j++)
	  sumPtTracksL25 = sumPtTracksL25 + isolationTracks[j]->pt();
	for(unsigned int j=0;j<signalTracks.size();j++)
	  sumPtTracksL25 = sumPtTracksL25 - signalTracks[j]->pt();

      }
    jetTracks = tauL3[i].jtaRef();
    math::XYZVector jetDirL3(jetTracks->first->px(),jetTracks->first->py(),jetTracks->first->pz());	
    int  trackIsolationL3 = (int)tauL3[i].discriminator(jetDirL3,matchingCone_, signalCone_, isolationCone_,1.,ptMin_,0);
    
    const TrackRef leadTkL3 = tauL3[i].leadingSignalTrack(jetDirL3,matchingCone_,1.);
    ptLeadTkL3=0.;
    nTracksL3 = 1000;
    if(!leadTkL3) 
      {}else{
	ptLeadTkL3 = (*leadTkL3).pt();
	const TrackRefVector signalTracks = tauL3[i].tracksInCone((*leadTkL25).momentum(), signalCone_, ptMin_ );
	const TrackRefVector isolationTracks = tauL3[i].tracksInCone((*leadTkL25).momentum(), isolationCone_,   ptMin_ );
	nTracksL3 = isolationTracks.size() - signalTracks.size();
	float sumPtTracksL3 = 0.;
	for(unsigned int j=0;j<isolationTracks.size();j++)
	  sumPtTracksL3 = sumPtTracksL3 + isolationTracks[j]->pt();
	for(unsigned int j=0;j<signalTracks.size();j++)
	  sumPtTracksL3 = sumPtTracksL3 - signalTracks[j]->pt();
      }

    HLTTau pippo(eta_,phi_,pt_,emIsol,trackIsolationL25,ptLeadTkL25,trackIsolationL3,ptLeadTkL3);
    pippo.setNL25TrackIsolation(nTracksL25);
    pippo.setNL3TrackIsolation(nTracksL3);
    pippo.setSumPtTracksL25(sumPtTracksL25);
    pippo.setSumPtTracksL3(sumPtTracksL3);
    pippo.setSeedEcalHitEt(l2info.seedEcalHitEt());
    pippo.setEcalClusterShape(l2info.ecalClusterShape());
    pippo.setNEcalHits(l2info.nEcalHits());		       
    pippo.setHcalIsolEt(l2info.hcalIsolEt());
    pippo.setSeedHcalHitEt(l2info.seedHcalHitEt());
    pippo.setHcalClusterShape(l2info.hcalClusterShape());
    pippo.setNHcalHits(l2info.nHcalHits());
    jetCollection->push_back(pippo);
      i++;
  }
  
  std::auto_ptr<reco::HLTTauCollection> selectedTaus(jetCollection);
  
  iEvent.put(selectedTaus);
  



}
