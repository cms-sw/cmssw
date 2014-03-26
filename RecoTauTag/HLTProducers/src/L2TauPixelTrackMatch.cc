
#include "RecoTauTag/HLTProducers/interface/L2TauPixelTrackMatch.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaPhi.h"


// all these debug printouts will need to be removed at some point
//#define DBG_PRINT(arg) (arg)
#define DBG_PRINT(arg) 


L2TauPixelTrackMatch::L2TauPixelTrackMatch(const edm::ParameterSet& conf)
{
  m_jetSrc	     = consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("JetSrc"));
  m_jetMinPt	     = conf.getParameter<double>("JetMinPt");
  m_jetMaxEta	     = conf.getParameter<double>("JetMaxEta");
  //m_jetMinN	     = conf.getParameter<int>("JetMinN");
  m_trackSrc	     = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackSrc"));
  m_trackMinPt	     = conf.getParameter<double>("TrackMinPt");
  m_deltaR	     = conf.getParameter<double>("deltaR");
  m_beamSpotTag      = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpotSrc"));

  produces<reco::CaloJetCollection>();
}


L2TauPixelTrackMatch::~L2TauPixelTrackMatch(){}


void L2TauPixelTrackMatch::produce(edm::Event& ev, const edm::EventSetup& es)
{
  using namespace std;
  using namespace reco;

  // *** Pick up beam spot ***
  
  // use beam spot for vertex x,y
  edm::Handle<BeamSpot> bsHandle;
  ev.getByToken( m_beamSpotTag, bsHandle);
  const reco::BeamSpot & bs = *bsHandle;
  math::XYZPoint beam_spot(bs.x0(), bs.y0(), bs.z0());
  DBG_PRINT(cout<<endl<<"beamspot "<<beam_spot<<endl);

  // *** Pick up pixel tracks ***

  edm::Handle<TrackCollection> tracksHandle;
  ev.getByToken(m_trackSrc,tracksHandle);

  // *** Pick up L2 tau jets that were previously selected by some other filter ***
  
  // first, get L2 object refs by the label
  edm::Handle<trigger::TriggerFilterObjectWithRefs> jetsHandle;
  ev.getByToken(m_jetSrc, jetsHandle);

  // now we can get pre-selected L2 tau jets
  std::vector<CaloJetRef> tau_jets;
  jetsHandle->getObjects(trigger::TriggerTau, tau_jets);
  const size_t n_jets = tau_jets.size();

  // *** Selects interesting tracks ***

  vector<TinyTrack> good_tracks;
  for(TrackCollection::const_iterator itrk = tracksHandle->begin(); itrk != tracksHandle->end(); ++itrk)
  {
    if (itrk->pt() < m_trackMinPt) continue;
    if ( std::abs(itrk->eta()) > m_jetMaxEta + m_deltaR ) continue;
       
    TinyTrack trk;
    trk.pt = itrk->pt();
    trk.phi = itrk->phi();
    trk.eta = itrk->eta();
    double dz = itrk->dz(beam_spot);
    trk.vtx = math::XYZPoint(bs.x(dz), bs.y(dz), dz);
    good_tracks.push_back(trk);
  }
  DBG_PRINT(cout<<"got "<<good_tracks.size()<<" good tracks;   "<<n_jets<<" tau jets"<<endl);


  // *** Match tau jets to intertesting tracks  and assign them new vertices ***

  // the new product
  std::auto_ptr<CaloJetCollection> new_tau_jets(new CaloJetCollection);
  int n_uniq = 0;
  if (good_tracks.size()) for (size_t i=0; i < n_jets; ++i)
  {
    reco::CaloJetRef jet = tau_jets[i];
    if ( jet->pt() < m_jetMinPt || std::abs(jet->eta()) > m_jetMaxEta ) continue;

    DBG_PRINT(cout<<i<<" :"<<endl);
    
    size_t n0 = new_tau_jets->size();
    
    for(vector<TinyTrack>::const_iterator itrk = good_tracks.begin(); itrk != good_tracks.end(); ++itrk)
    {
      DBG_PRINT(cout<<"  trk pt,eta,phi,z: "<<itrk->pt<<" "<<itrk->eta<<" "<<itrk->phi<<" "<<itrk->vtx.z()<<" \t\t ");

      math::XYZTLorentzVector new_jet_dir = Jet::physicsP4(itrk->vtx, *jet, itrk->vtx);
      float dphi = reco::deltaPhi(new_jet_dir.phi(), itrk->phi);
      float deta = new_jet_dir.eta() - itrk->eta;
      
      DBG_PRINT(cout<<" jet pt,deta,dphi,dr: "<<jet->pt()<<" "<<deta<<" "<<dphi<<" "<<sqrt(dphi*dphi + deta*deta)<<endl);
      
      if ( dphi*dphi + deta*deta > m_deltaR*m_deltaR ) continue;
      
      DBG_PRINT(cout<<"  jet-trk match!"<<endl);
      
      // create a jet copy and assign a new vertex to it
      CaloJet new_jet = *jet;
      new_jet.setVertex(itrk->vtx);
      
      new_tau_jets->push_back(new_jet);
    }
    DBG_PRINT(cout<<"  nmatchedjets "<<new_tau_jets->size() - n0<<endl);
    if (new_tau_jets->size() - n0 > 0 ) n_uniq++;
    
    ///if (jet_with_vertices.size()) new_tau_jets->push_back(jet_with_vertices);
  }
  DBG_PRINT(cout<<"n_uniq_matched_jets "<<n_uniq<<endl<<"storing njets "<<new_tau_jets->size()<<endl);
  
  // store the result
  ev.put(new_tau_jets);
}
