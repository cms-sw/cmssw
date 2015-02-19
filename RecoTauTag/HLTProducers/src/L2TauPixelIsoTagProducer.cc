// 
// Original Author:  Vadim Khotilovich
//         Created:  2012-03-07
//

#include "RecoTauTag/HLTProducers/interface/L2TauPixelIsoTagProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


L2TauPixelIsoTagProducer::L2TauPixelIsoTagProducer(const edm::ParameterSet& conf):
  m_jetSrc_token( consumes<edm::View<reco::Jet> >(conf.getParameter<edm::InputTag>("JetSrc") ) ),
  m_vertexSrc_token( consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("VertexSrc") ) ),
  m_trackSrc_token( consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackSrc") ) ),  // for future use (now tracks are taken directly from PV)
  m_beamSpotSrc_token( consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpotSrc") ) ),
  m_maxNumberPV( conf.getParameter<int>("MaxNumberPV") ), // for future use, now is assumed to be = 1
  m_trackMinPt( conf.getParameter<double>("TrackMinPt") ),
  m_trackMaxDxy( conf.getParameter<double>("TrackMaxDxy") ),
  m_trackMaxNChi2( conf.getParameter<double>("TrackMaxNChi2") ),
  m_trackMinNHits( conf.getParameter<int>("TrackMinNHits") ),
  m_trackPVMaxDZ( conf.getParameter<double>("TrackPVMaxDZ") ), // for future use with tracks not from PV
  m_isoCone2Min( std::pow(conf.getParameter<double>("IsoConeMin"), 2) ),
  m_isoCone2Max( std::pow(conf.getParameter<double>("IsoConeMax"), 2) )
{
  produces<reco::JetTagCollection>(); 
}


void L2TauPixelIsoTagProducer::produce(edm::StreamID sid, edm::Event& ev, const edm::EventSetup& es) const
{
  using namespace reco;
  using namespace std;
  using namespace edm;
  //m_trackSrc.encode();

  edm::Handle<BeamSpot> bs;
  ev.getByToken( m_beamSpotSrc_token, bs);


  // Get jets
  Handle< View<Jet> > jets_h;
  ev.getByToken (m_jetSrc_token, jets_h);
  vector<RefToBase<Jet> > jets;
  jets.reserve (jets_h->size());
  for (size_t i = 0; i < jets_h->size(); ++i)  jets.push_back( jets_h->refAt(i) );


  // define the product to store
  auto_ptr<JetTagCollection> jetTagCollection;
  if (jets.empty())
  {
    jetTagCollection.reset( new JetTagCollection() );
  }
  else
  {
    jetTagCollection.reset( new JetTagCollection( RefToBaseProd<Jet>( *jets_h.product() ) ) );
  }
  // by default, initialize all the jets as isolated:
  for (const auto &jet : jets)  (*jetTagCollection)[jet] = 0.f;


  // Get pixel vertices (their x,y positions are already supposed to be defined from the BeamSpot)
  Handle<VertexCollection> vertices;
  ev.getByToken(m_vertexSrc_token, vertices);

  // find the primary vertex (the 1st valid non-fake vertex in the collection)
  const Vertex *pv = 0;
  for(const auto & v : *(vertices.product()) )
  {
    if(!v.isValid() || v.isFake()) continue;
    pv = &v;
    break;
  }

  // If primary vertex exists, calculate jets' isolation:
  if(pv && jets.size())
  {
    for (const auto & jet : jets)
    {
      // re-calculate jet eta in PV:
      float jet_eta = Jet::physicsEta(pv->z(), jet->eta());
      float jet_phi = jet->phi();

      // to calculate isolation, use only tracks that were assigned to the vertex
      float iso = 0.f;
      for(vector<TrackBaseRef>::const_iterator tr = pv->tracks_begin(); tr != pv->tracks_end(); ++tr)
      {
        if ((*tr)->pt() < m_trackMinPt) continue;
        if ((*tr)->numberOfValidHits() < m_trackMinNHits) continue;
        if ((*tr)->normalizedChi2() > m_trackMaxNChi2) continue;
        if (std::abs( (*tr)->dxy(*bs) ) > m_trackMaxDxy) continue;

        float dr2 = deltaR2 (jet_eta, jet_phi, (*tr)->eta(), (*tr)->phi());

        if (dr2 >= m_isoCone2Min && dr2 <= m_isoCone2Max) iso += 1.;
      }

      (*jetTagCollection)[jet] = iso;
    }
  }

  ev.put(jetTagCollection);
}

void L2TauPixelIsoTagProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("JetSrc",edm::InputTag("hltL2DiTauCaloJets"))->setComment("Jet source collection");
  desc.add<edm::InputTag>("BeamSpotSrc",edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("VertexSrc",edm::InputTag("hltPixelVertices"))->setComment("Collection of vertices where isolation tracks come from");
  desc.add<int>("MaxNumberPV",1)->setComment("No. of considered vertices (not used yet)");
  desc.add<double>("IsoConeMax",0.4)->setComment("Outer radius of isolation annulus");
  desc.add<double>("IsoConeMin",0.2)->setComment("Inner radius of isolation annulus");
  desc.add<double>("TrackMinPt",1.6)->setComment("Isolation track quality: min. pT");
  desc.add<int>("TrackMinNHits",3)->setComment("Isolation track quality: min. no. of hits");
  desc.add<double>("TrackMaxNChi2",100.0)->setComment("Isolation track quality: max. chi2/ndof");
  desc.add<double>("TrackPVMaxDZ",0.1)->setComment("Isolation track quality: max. dz");;
  desc.add<double>("TrackMaxDxy",0.2)->setComment("Isolation track quality: max. dxy");;
  desc.add<edm::InputTag>("TrackSrc",edm::InputTag(""))->setComment("Not used yet");
  descriptions.setComment("Produces isolation tag for caloJets/L2Taus");
  descriptions.add("L2TauPixelIsoTagProducer",desc);

}
