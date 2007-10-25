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
  rmax_ = iConfig.getParameter<double>("EcalIsoRMax");
  rmin_ = iConfig.getParameter<double>("EcalIsoRMin");
  matchingCone_ = iConfig.getParameter<double>("MatchingCone");
  ptMinLeadTk_ = iConfig.getParameter<double>("PtLeadTk");

  produces<reco::HLTTauCollection>();
}

HLTTauProducer::~HLTTauProducer(){ }

void HLTTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  

  HLTTauCollection * jetCollection = new HLTTauCollection;
 
  edm::Handle<EMIsolatedTauTagInfoCollection> tauL2Jets;
  iEvent.getByLabel(emIsolatedJetsL2_ , tauL2Jets );

  edm::Handle<IsolatedTauTagInfoCollection> tauL25Jets;
  iEvent.getByLabel(trackIsolatedJetsL25_, tauL25Jets );
  
  edm::Handle<IsolatedTauTagInfoCollection> tauL3Jets;
  iEvent.getByLabel(trackIsolatedJetsL3_, tauL3Jets );

  EMIsolatedTauTagInfoCollection tauL2 = *(tauL2Jets.product());
  IsolatedTauTagInfoCollection tauL25 = *(tauL25Jets.product());
  IsolatedTauTagInfoCollection tauL3 = *(tauL3Jets.product());
  
  double metCut_= 1000.;
  for(unsigned int i=0 ;i !=tauL2.size(); i++ ) {
    double emIsol  = tauL2[i].pIsol(rmax_,rmin_);
    int trackIsolationL25 = (int)tauL25[i].discriminator();
    const TrackRef leadTkL25 = tauL25[i].leadingSignalTrack(matchingCone_, ptMinLeadTk_);
    double ptLeadTkL25=0.;
    
    if(!leadTkL25) 
      {}else{
	ptLeadTkL25 = (*leadTkL25).pt();
      }
	

    int  trackIsolationL3 = (int)tauL3[i].discriminator();
    const TrackRef leadTkL3 = tauL3[i].leadingSignalTrack(matchingCone_, ptMinLeadTk_);
    double ptLeadTkL3=0.;
    if(!leadTkL3) 
      {}else{
	ptLeadTkL3 = (*leadTkL3).pt();
      }

    HLTTau pippo(metCut_,emIsol,trackIsolationL25,ptLeadTkL25,trackIsolationL3,ptLeadTkL3);
      jetCollection->push_back(pippo);
  }
  
  auto_ptr<reco::HLTTauCollection> selectedTaus(jetCollection);
  
  iEvent.put(selectedTaus);
  



}
