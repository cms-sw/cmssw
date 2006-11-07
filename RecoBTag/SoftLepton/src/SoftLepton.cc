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
// $Id: SoftLepton.cc,v 1.3 2006/10/31 02:53:09 fwyzard Exp $
//


#include <memory>
#include <string>
#include <iostream>
using namespace std;

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/SoftLepton.h"

#include "RecoBTag/SoftLepton/src/LeptonTaggerBase.h"
#include "RecoBTag/SoftLepton/src/MuonTagger.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerNoIP.h"

using namespace reco;

reco::Vertex s_nominalBeamSpot(  );

SoftLepton::SoftLepton(const edm::ParameterSet& iConfig) :
  m_config( iConfig ),
  m_jetTracksAssociator(   iConfig.getParameter<std::string>( "jetTracks"          ) ),
  m_primaryVertexProducer( iConfig.getParameter<std::string>( "primaryVertex"      ) ),
  m_leptonProducer(        iConfig.getParameter<std::string>( "leptons"            ) ),
  m_outputInstanceName(    iConfig.getParameter<std::string>( "outputInstanceName" ) ),
//m_algo(                  iConfig.getParameter<edm::ParameterSet>( "AlgorithmPSet" ) )
  m_algo()
{
  produces<reco::JetTagCollection>( m_outputInstanceName );     // several producers - use a label
  produces<reco::SoftLeptonTagInfoCollection>();                // only one producer

  reco::Vertex::Point p( 0, 0, 0 );
  reco::Vertex::Error e;
  e(0,0) = 0.0015 * 0.0015;
  e(1,1) = 0.0015 * 0.0015;
  e(2,2) = 15. * 15.;
  m_nominalBeamSpot = new reco::Vertex( p, e, 1, 1, 0 );

  m_algo.setDeltaRCut( m_config.getParameter<double>("deltaRCut") );
  m_algo.refineJetAxis( m_config.getParameter<unsigned int>("refineJetAxis") );
}


SoftLepton::~SoftLepton() {
  delete m_nominalBeamSpot;
}

void
SoftLepton::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // input objects
  Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
  iEvent.getByLabel(m_jetTracksAssociator, jetTracksAssociation);

  Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByLabel(m_primaryVertexProducer, primaryVertex);

  Handle<reco::MuonCollection> leptons;
  iEvent.getByLabel(m_leptonProducer, leptons);

  // output collections
  std::auto_ptr<reco::JetTagCollection>            baseCollection( new reco::JetTagCollection() );
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> extCollection(  new reco::SoftLeptonTagInfoCollection() );

  const reco::Vertex * pv = primaryVertex->size() ? &(*primaryVertex->begin()) : m_nominalBeamSpot;

  #ifdef DEBUG
  std::cerr << std::endl;
  std::cerr << "Found " << jetTracksAssociation->size() << " jet with tracks:" << std::endl;
  #endif // DEBUG
  for (reco::JetTracksAssociationCollection::const_iterator j = jetTracksAssociation->begin();
       j !=jetTracksAssociation->end();
       ++j) {
    unsigned int i = j->key.key();
  //for (unsigned int i = 0; i < jetTracksAssociation->size(); i++) {
    reco::JetTracksAssociationRef jetRef( jetTracksAssociation, i );
    std::pair<reco::JetTag, reco::SoftLeptonTagInfo> result = m_algo.tag( jetRef, *pv, *leptons );
    baseCollection->push_back( result.first );
    extCollection->push_back( result.second );
  }

  edm::OrphanHandle<reco::JetTagCollection> handleJetTag = iEvent.put( baseCollection, m_outputInstanceName );
  for (unsigned int i = 0; i < extCollection->size(); i++)
    (*extCollection)[i].setJetTag( reco::JetTagRef( handleJetTag, i ) );
  iEvent.put(extCollection);
}

// ------------ method called once each job just before starting event loop  ------------
void 
SoftLepton::beginJob(const edm::EventSetup& iSetup) {
  // grab a TransientTrack helper from the Event Setup
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get( "TransientTrackBuilder", builder );
  m_algo.setTransientTrackBuilder( builder.product() );

  // FIXME: this should become something in the EventSetup, too
  LeptonTaggerBase * tagger = new MuonTagger();
  m_algo.setConcreteTagger( tagger );
  // FIXME: tagger is never deleted, but who cares?
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SoftLepton::endJob(void) {
}

// define this as a plug-in
DEFINE_FWK_MODULE(SoftLepton)
