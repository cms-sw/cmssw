#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "testLeptonAssociator.h"

using namespace reco;
using namespace std;
using namespace edm;

class TrackAssociatorByHits; 
class TrackerHitAssociator; 

testLeptonAssociator::testLeptonAssociator(edm::ParameterSet const& iConfig) {
  m_trackAssociator = iConfig.getParameter<std::string>(   "trackAssociator" );
  m_recoTracks      = iConfig.getParameter<edm::InputTag>( "recoTracks" );
  m_recoLeptons     = iConfig.getParameter<edm::InputTag>( "recoLeptons" );
  m_trackingTruth   = iConfig.getParameter<edm::InputTag>( "trackingTruth" );
  m_flavour         = iConfig.getParameter<unsigned int>(  "leptonFlavour" );
}

testLeptonAssociator::~testLeptonAssociator() {
}

void testLeptonAssociator::beginJob(const EventSetup & iSetup) {
  edm::ESHandle<TrackAssociatorBase> associatorHandle;
  iSetup.get<TrackAssociatorRecord>().get(m_trackAssociator, associatorHandle);
  m_associator = associatorHandle.product();
}

void testLeptonAssociator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<reco::TrackCollection> recoTrackHandle;
  iEvent.getByLabel(m_recoTracks, recoTrackHandle);
  const reco::TrackCollection& recoTrackCollection = *(recoTrackHandle.product()); 
  
  edm::Handle<reco::TrackCollection> recoLeptonHandle;
  iEvent.getByLabel(m_recoLeptons, recoLeptonHandle);
  const reco::TrackCollection& recoLeptonCollection = *(recoLeptonHandle.product()); 
  
  edm::Handle<TrackingParticleCollection> trackingParticleHandle ;
  iEvent.getByLabel(m_trackingTruth, trackingParticleHandle);
  const TrackingParticleCollection& trackingParticleCollection = *(trackingParticleHandle.product());

  /* TODO
   * tracking efficiency:                   associate sim-leptons to reco-tracks
   * tracking + identification efficiency:  associate sim-leptons to reco-leptons
   * fake-rate / misid:                     associate (unassociated) reco-leptons to (non-leptons) sim-tracks
   */
  
  cout << fixed;

  cout << endl;
  cout << "Found " << setw(6) << trackingParticleCollection.size() << " TrackingParticles" << flush;
  unsigned int count = 0;
  for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i)
    if (abs(trackingParticleCollection[i].pdgId()) == m_flavour)
      ++count;
  cout << " ( " << setw(2) << count << " leptons )" << endl;
  cout << "      " << setw(6) << recoTrackCollection.size()        << " Tracks" << endl;
  cout << "      " << setw(6) << recoLeptonCollection.size()       << " Leptons" << endl;

  // look for tracks and muons associated to the tracking particles
  {
    reco::SimToRecoCollection map_tracks  = m_associator->associateSimToReco(recoTrackHandle, trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection map_leptons = m_associator->associateSimToReco(recoLeptonHandle, trackingParticleHandle, &iEvent );
    for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i) {
      TrackingParticleRef tp (trackingParticleHandle, i);
      if (abs(tp->pdgId()) != m_flavour) continue;
      cout << "--> TrackingParticle" 
           << " ["     << setw(4) << tp.index() << "]"
           << " type:" << setw(6) << tp->pdgId() 
           << " pT: "  << setw(6) << setprecision(3) << tp->pt() 
           << " eta: " << setw(6) << setprecision(3) << tp->eta() 
           << endl;
      if (map_tracks.find(tp) != map_tracks.end()) { 
        const reco::SimToRecoCollection::result_type & tracks = map_tracks[tp];
        // cout << " matched to " << setw(2) << right << tracks.size() << " Tracks" << std::endl;
        for (std::vector<std::pair<TrackRef, double> >::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
          TrackRef track = it->first;
          double quality = it->second;
          cout << "    Track  "
               << " {"     << setw(2) << track->found() << "}    "
               << " ["     << setw(4) << track.index() << "]"
               << "            "
               << " pT: "  << setw(6) << setprecision(3) << track->pt()
               << " eta: " << setw(6) << setprecision(3) << track->eta()
               << " quality: " << quality 
               << endl;
        }
      } else {
        // cout << " matched to no Tracks" << std::endl;
      }
      if (map_leptons.find(tp) != map_leptons.end()) { 
        const reco::SimToRecoCollection::result_type & leptons = map_leptons[tp];
        // cout << " matched to " << setw(2) << right << leptons.size() << " Leptons" << std::endl;
        for (std::vector<std::pair<TrackRef, double> >::const_iterator it = leptons.begin(); it != leptons.end(); ++it) {
          TrackRef lepton = it->first;
          double quality = it->second;
          cout << "    Lepton "
               << " {"     << setw(2) << lepton->found() << "}    "
               << " ["     << setw(4) << lepton.index() << "]"
               << "            "
               << " pT: "  << setw(6) << setprecision(3) << lepton->pt()
               << " eta: " << setw(6) << setprecision(3) << lepton->eta()
               << " quality: " << quality 
               << endl;
        }
      } else {
        // cout << " matched to no Leptons" << std::endl;
      }
    }

  }

  // look for tracking particles associated to the reconstructed muons
  {
    reco::RecoToSimCollection map = m_associator->associateRecoToSim (recoLeptonHandle, trackingParticleHandle, &iEvent );
    for (TrackCollection::size_type i = 0; i < recoLeptonCollection.size(); ++i) {
      TrackRef lepton(recoLeptonHandle, i);
          cout << "<-- Lepton "
               << " {"     << setw(2) << lepton->found() << "}    "
               << " ["     << setw(4) << lepton.index() << "]"
               << "            "
               << " pT: "  << setw(6) << setprecision(3) << lepton->pt()
               << " eta: " << setw(6) << setprecision(3) << lepton->eta()
               << endl;
      if (map.find(lepton) != map.end()) {
        reco::RecoToSimCollection::result_type particles = map[lepton];
        // cout << " matched to " << setw(2) << right << particles.size() << " TrackingParticles" << std::endl;
        for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = particles.begin(); it != particles.end(); ++it) {
          TrackingParticleRef tp = it->first;
          double quality = it->second;
          cout << "    TrackingParticle" 
               << " ["     << setw(4) << tp.index() << "]"
               << " type:" << setw(6) << tp->pdgId() 
               << " pT: "  << setw(6) << setprecision(3) << tp->pt() 
               << " eta: " << setw(6) << setprecision(3) << tp->eta() 
               << " quality: " << quality 
               << endl;
        }
      } else {
        // cout << " matched to  0 TrackingParticles" << endl;
      }
    }

  }

}

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testLeptonAssociator);
