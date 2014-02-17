// 
// Original Author:  Vadim Khotilovich
//         Created:  2012-03-07
// $Id: L2TauPixelIsoTagProducer.cc,v 1.1 2012/03/13 16:21:42 khotilov Exp $
//

#include "RecoTauTag/HLTProducers/interface/L2TauPixelIsoTagProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"


L2TauPixelIsoTagProducer::L2TauPixelIsoTagProducer(const edm::ParameterSet& conf)
{
  m_jetSrc      = conf.getParameter<edm::InputTag>("JetSrc");
  m_vertexSrc   = conf.getParameter<edm::InputTag>("VertexSrc");
  m_trackSrc    = conf.getParameter<edm::InputTag>("TrackSrc");  // for future use (now tracks are taken directly from PV)
  m_beamSpotSrc = conf.getParameter<edm::InputTag>("BeamSpotSrc");

  m_maxNumberPV = conf.getParameter<int>("MaxNumberPV"); // for future use, now is assumed to be = 1

  m_trackMinPt    = conf.getParameter<double>("TrackMinPt");
  m_trackMaxDxy   = conf.getParameter<double>("TrackMaxDxy");
  m_trackMaxNChi2 = conf.getParameter<double>("TrackMaxNChi2");
  m_trackMinNHits = conf.getParameter<int>("TrackMinNHits");
  m_trackPVMaxDZ    = conf.getParameter<double>("TrackPVMaxDZ"); // for future use with tracks not from PV

  m_isoCone2Min  = std::pow(conf.getParameter<double>("IsoConeMin"), 2);
  m_isoCone2Max  = std::pow(conf.getParameter<double>("IsoConeMax"), 2);

  produces<reco::JetTagCollection>(); 
}


void L2TauPixelIsoTagProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  using namespace reco;
  using namespace std;
  using namespace edm;
  m_trackSrc.encode();

  edm::Handle<BeamSpot> bs;
  ev.getByLabel( m_beamSpotSrc, bs);


  // Get jets
  Handle< View<Jet> > jets_h;
  ev.getByLabel (m_jetSrc, jets_h);
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
  ev.getByLabel(m_vertexSrc, vertices);

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
