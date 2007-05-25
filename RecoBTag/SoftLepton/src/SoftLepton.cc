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
// $Id: SoftLepton.cc,v 1.17 2007/05/11 11:29:04 fwyzard Exp $
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/SoftLepton/interface/SoftLepton.h"

using namespace std;
using namespace edm;
using namespace reco;

const reco::Vertex SoftLepton::s_nominalBeamSpot(
  reco::Vertex::Point( 0, 0, 0 ),
  reco::Vertex::Error( ROOT::Math::SVector<double,6>( 0.0015 * 0.0015, //          0.0,        0.0
                                                                  0.0, 0.0015 * 0.0015, //     0.0  
                                                                  0.0,             0.0, 15. * 15. ) ),
  1, 1, 0 );

SoftLepton::SoftLepton(const edm::ParameterSet& iConfig) :
  m_config( iConfig ),
  m_concreteTagger(        iConfig.getParameter<std::string>( "leptonTagger"       ) ), 
  m_jetTracksAssociator(   iConfig.getParameter<std::string>( "jetTracks"          ) ),
  m_primaryVertexProducer( iConfig.getParameter<std::string>( "primaryVertex"      ) ),
  m_leptonProducer(        iConfig.getParameter<std::string>( "leptons"            ) ),
  m_algo()
{
  produces<reco::JetTagCollection>();
  produces<reco::SoftLeptonTagInfoCollection>();

  m_algo.setDeltaRCut( m_config.getParameter<double>("deltaRCut") );
  m_algo.refineJetAxis( m_config.getParameter<unsigned int>("refineJetAxis") );
}

SoftLepton::~SoftLepton() {
}

void
SoftLepton::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // input objects
  Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
  iEvent.getByLabel(m_jetTracksAssociator, jetTracksAssociation);

  Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByLabel(m_primaryVertexProducer, primaryVertex);

  reco::TrackRefVector leptons;
  // try to access the input collection as a collection of Electons, Muons or Tracks
  // FIXME: it would be nice not to have to rely on exceptions
  try {
    Handle<reco::ElectronCollection> h_electrons;
    iEvent.getByLabel(m_leptonProducer, h_electrons);
    #ifdef DEBUG
    cerr << "SoftLepton::produce : collection " << m_leptonProducer << " found, identified as ElectronCollection" << endl;
    #endif
    for (reco::ElectronCollection::const_iterator electron = h_electrons->begin(); electron != h_electrons->end(); ++electron)
      leptons.push_back(electron->track());
  } 
  catch(edm::Exception e)  { 
    // electrons not found, look for muons
    try {
      Handle<reco::MuonCollection> h_muons;
      iEvent.getByLabel(m_leptonProducer, h_muons);
      #ifdef DEBUG
      cerr << "SoftLepton::produce : collection " << m_leptonProducer << " found, identified as MuonCollection" << endl;
      #endif
      for (reco::MuonCollection::const_iterator muon = h_muons->begin(); muon != h_muons->end(); ++muon)
        if(! muon->combinedMuon().isNull() )
          leptons.push_back( muon->combinedMuon() );
        else 
          cerr << "SoftLepton::produce : found a Null edm::Ref in MuonCollection " << m_leptonProducer << ", skipping it" << endl;
    }
    catch(edm::Exception e) {
      // electrons or muons not found, look for tracks
      Handle<reco::TrackCollection> h_tracks;
      iEvent.getByLabel(m_leptonProducer, h_tracks);
      #ifdef DEBUG
      cerr << "SoftLepton::produce : collection " << m_leptonProducer << " found, identified as TrackCollection" << endl;
      #endif
      for (unsigned int i = 0; i < h_tracks->size(); i++)
        leptons.push_back( TrackRef(h_tracks, i) );
    }
  }

  // output collections
  std::auto_ptr<reco::JetTagCollection>            baseCollection( new reco::JetTagCollection() );
  std::auto_ptr<reco::SoftLeptonTagInfoCollection> extCollection(  new reco::SoftLeptonTagInfoCollection() );

  reco::Vertex pv;
  if (primaryVertex->size()) {
    PrimaryVertexSorter pvs;
    pv = pvs.sortedList(*(primaryVertex.product())).front();
  } else {
    pv = s_nominalBeamSpot;
  }

  #ifdef DEBUG
  std::cerr << std::endl;
  std::cerr << "Found " << jetTracksAssociation->size() << " jet with tracks:" << std::endl;
  #endif // DEBUG
  for (unsigned int i = 0; i < jetTracksAssociation->size(); ++i) {
    reco::JetTracksAssociationRef jetRef( jetTracksAssociation, i );
    std::pair<reco::JetTag, reco::SoftLeptonTagInfo> result = m_algo.tag( jetRef, pv, leptons );
    #ifdef DEBUG
    std::cerr << "  Jet " << std::setw(2) << i << " has " << std::setw(2) << result.first.tracks().size() << " tracks and " << std::setw(2) << result.second.leptons() << " leptons" << std::endl;
    std::cerr << "  Tagger result: " << result.first.discriminator() << endl;
    #endif // DEBUG
    baseCollection->push_back( result.first );
    extCollection->push_back( result.second );
  }

  // the base collection needs a link to the extended collection
  edm::OrphanHandle<reco::SoftLeptonTagInfoCollection> extHandle = iEvent.put( extCollection );
  for (unsigned int i = 0; i < baseCollection->size(); i++)
    (*baseCollection)[i].setTagInfo( edm::RefToBase<BaseTagInfo>( reco::SoftLeptonTagInfoRef( extHandle, i) ) );
  iEvent.put(baseCollection);
}

// ------------ method called once each job just before starting event loop  ------------
void 
SoftLepton::beginJob(const edm::EventSetup& iSetup) {
  // grab a TransientTrack helper from the Event Setup
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get( "TransientTrackBuilder", builder );
  m_algo.setTransientTrackBuilder( builder.product() );

  // grab the concrete soft lepton b tagger from the Event Setup
  edm::ESHandle<JetTagComputer> tagger;
  iSetup.get<JetTagComputerRecord>().get( m_concreteTagger, tagger );
  m_algo.setConcreteTagger( tagger.product() );
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SoftLepton::endJob(void) {
}

