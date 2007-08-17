// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLepton
// 
/**\class SoftLepton SoftLepton.cc RecoBTag/SoftLepton/src/SoftLepton.cc

 Description: CMSSW EDProducer wrapper for sot lepton b tagging.

 Implementation:
     The actual tagging is performed by SoftLeptonAlgorithm.
*/
//
// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006
// $Id: SoftLepton.cc,v 1.1 2007/08/17 23:10:10 fwyzard Exp $
//


#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "findProductIDByLabel.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "SoftLepton.h"

using namespace std;
using namespace edm;
using namespace reco;

const reco::Vertex SoftLepton::s_nominalBeamSpot(
  reco::Vertex::Point( 0, 0, 0 ),
  reco::Vertex::Error( ROOT::Math::SVector<double,6>( 0.0015 * 0.0015, //          0.0,        0.0
                                                                  0.0, 0.0015 * 0.0015, //     0.0  
                                                                  0.0,             0.0, 15. * 15. ) ),
  1, 1, 0 );

SoftLepton::SoftLepton(const edm::ParameterSet & iConfig) :
  m_jets(          iConfig.getParameter<edm::InputTag>( "jets" ) ),
  m_primaryVertex( iConfig.getParameter<edm::InputTag>( "primaryVertex" ) ),
  m_leptons(       iConfig.getParameter<edm::InputTag>( "leptons" ) ),
  m_algo(          iConfig.getParameter<edm::ParameterSet> ( "algorithmConfiguration") ),
  m_quality(       iConfig.getParameter<double>( "leptonQuality" ) )
{
  produces<reco::SoftLeptonTagInfoCollection>();
}

SoftLepton::~SoftLepton() {
}

void
SoftLepton::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // input objects

  // input jets (and possibly tracks)
  ProductID jets_id;
  std::vector<edm::RefToBase<reco::Jet> > jets;
  std::vector<reco::TrackRefVector>       tracks;
  if (jets_id = edm::findProductIDByLabel<reco::JetTracksAssociationCollection>(iEvent, m_jets), jets_id.isValid())
  {
    Handle<reco::JetTracksAssociationCollection> h_jtas;
    iEvent.get(jets_id, h_jtas);

    unsigned int size = h_jtas->size();
    jets.resize(size);
    tracks.resize(size);
    for (unsigned int i = 0; i < size; ++i) {
      jets[i]   = (*h_jtas)[i].first;
      tracks[i] = (*h_jtas)[i].second;
    }
  }
  else
  if (jets_id = edm::findProductIDByLabel<reco::CaloJetCollection>(iEvent, m_jets), jets_id.isValid())
  {
    Handle<reco::CaloJetCollection> h_jets;
    iEvent.get(jets_id, h_jets);

    unsigned int size = h_jets->size();
    jets.resize(size);
    tracks.resize(size);
    for (unsigned int i = 0; i < h_jets->size(); i++)
      jets[i] = edm::RefToBase<reco::Jet>( reco::CaloJetRef(h_jets, i) );
  }
  else
  {
    throw edm::Exception(edm::errors::NotFound) << "Object " << m_jets << " of type among (\"reco::JetTracksAssociationCollection\", \"reco::CaloJetCollection\") not found";
  }
  
  // input primary vetex (optional, can be "none")
  reco::Vertex vertex;
  Handle<reco::VertexCollection> h_primaryVertex;
  if (m_primaryVertex.label() == "nominal") {
    vertex = s_nominalBeamSpot;
  } else
  if (m_primaryVertex.label() == "beamSpot") {
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByType(recoBeamSpotHandle);
    vertex = reco::Vertex(recoBeamSpotHandle->position(), recoBeamSpotHandle->covariance3D(), 1, 1, 0);
  } else {
    iEvent.getByLabel(m_primaryVertex, h_primaryVertex);
    if (h_primaryVertex->size()) {
      PrimaryVertexSorter pvs;
      // FIXME is this still needed in 1.5.x ?
      vertex = pvs.sortedList(*(h_primaryVertex.product())).front();
    } else {
      // fall back to nominal beam spot
      vertex = s_nominalBeamSpot;
    }
  }

  // input leptons (can be of different types)
  ProductID leptons_id;
  std::vector<edm::RefToBase<reco::Track> > leptons;
  // try to access the input collection as a collection of Electrons, Muons or Tracks
  // look for Electrons
  if (leptons_id = edm::findProductIDByLabel<reco::ElectronCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::ElectronCollection> h_electrons;
    iEvent.get(leptons_id, h_electrons);
    for (reco::ElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
      leptons.push_back(edm::RefToBase<reco::Track>( electron->track() ));
  }
  else
  // look for PixelMatchElectrons
  if (leptons_id = edm::findProductIDByLabel<reco::PixelMatchElectronCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::PixelMatchElectronCollection> h_electrons;
    iEvent.get(leptons_id, h_electrons);
    for (reco::PixelMatchElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
      leptons.push_back(edm::RefToBase<reco::Track>( electron->track() ));
  } 
  else
  // look for PixelMatchGsfElectrons
  if (leptons_id = edm::findProductIDByLabel<reco::PixelMatchGsfElectronCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::PixelMatchGsfElectronCollection> h_electrons;
    iEvent.get(leptons_id, h_electrons);
    for (reco::PixelMatchGsfElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
      leptons.push_back(edm::RefToBase<reco::Track>( electron->gsfTrack() ));
  } 
  else
  // electrons not found, look for muons
  if (leptons_id = edm::findProductIDByLabel<reco::MuonCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::MuonCollection> h_muons;
    iEvent.get(leptons_id, h_muons);
    for (reco::MuonCollection::const_iterator muon = h_muons->begin(); muon != h_muons->end(); ++muon)
    {
      if (! muon->combinedMuon().isNull() and muon->getCaloCompatibility() > m_quality)
        leptons.push_back(edm::RefToBase<reco::Track>( muon->combinedMuon() ));
      else 
      if (! muon->track().isNull() and muon->getCaloCompatibility() > m_quality)
        leptons.push_back(edm::RefToBase<reco::Track>( muon->track() ));
    }
  }
  else
  // look for GsfTracks
  if (leptons_id = edm::findProductIDByLabel<reco::GsfTrackCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::GsfTrackCollection> h_tracks;
    iEvent.get(leptons_id, h_tracks);
    for (unsigned int i = 0; i < h_tracks->size(); i++)
      leptons.push_back(edm::RefToBase<reco::Track>( reco::GsfTrackRef(h_tracks, i) ));
  }
  else
  // look for Tracks
  if (leptons_id = edm::findProductIDByLabel<reco::TrackCollection>(iEvent, m_leptons), leptons_id.isValid())
  {
    Handle<reco::TrackCollection> h_tracks;
    iEvent.get(leptons_id, h_tracks);
    for (unsigned int i = 0; i < h_tracks->size(); i++)
      leptons.push_back(edm::RefToBase<reco::Track>( reco::TrackRef(h_tracks, i) ));
  }
  else
  {
    throw edm::Exception(edm::errors::NotFound) << "Object " << m_leptons << " of type among (\"reco::ElectronCollection\", \"reco::PixelMatchElectronCollection\", \"reco::PixelMatchGsfElectronCollection\", \"reco::MuonCollection\", \"reco::GsfTrackCollection\", \"reco::TrackCollection\") not found";
  }

  // output collections
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> outputCollection(  new reco::SoftLeptonTagInfoCollection() );
  for (unsigned int i = 0; i < jets.size(); ++i) {
    reco::SoftLeptonTagInfo result = m_algo.tag( jets[i], tracks[i], leptons, vertex );
    outputCollection->push_back( result );
  }
  iEvent.put( outputCollection );
}

// ------------ method called once each job just before starting event loop  ------------
void 
SoftLepton::beginJob(const edm::EventSetup& iSetup) {
  // grab a TransientTrack helper from the Event Setup
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get( "TransientTrackBuilder", builder );
  m_algo.setTransientTrackBuilder( builder.product() );
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SoftLepton::endJob(void) {
  m_algo.setTransientTrackBuilder( NULL );
}

