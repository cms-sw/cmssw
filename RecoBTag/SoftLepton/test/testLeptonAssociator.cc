#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <set>

#include <boost/tuple/tuple.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

class testLeptonAssociator : public edm::EDAnalyzer {
public:
  explicit testLeptonAssociator(const edm::ParameterSet& iConfig);
  virtual ~testLeptonAssociator();
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  edm::InputTag m_recoTracks;
  edm::InputTag m_standAloneMuons;
  edm::InputTag m_globalMuons;
  edm::InputTag m_trackingTruth;
  unsigned int  m_flavour;
  double        m_ptcut;

  const TrackAssociatorBase* m_associatorByHits;
  const TrackAssociatorBase* m_associatorByChi2;
};


std::ostream& operator<< (std::ostream& out, TrackRef ref) {
  out << " {"     << setw(2) << ref->found() << "}    "
      << " ["     << setw(4) << ref.index() << "]"
      << "            "
      << " pT: "  << setw(6) << setprecision(3) << ref->pt()
      << " eta: " << setw(6) << setprecision(3) << ref->eta()
      << " phi: " << setw(6) << setprecision(3) << ref->phi();
  return out;
}

std::ostream& operator<< (std::ostream& out, MuonRef ref) {
  out << " {"     << setw(2) << ref->track()->found() << "+" << setw(2) << ref->standAloneMuon()->found() << "} "
      << " ["     << setw(4) << ref.index() << "]"
      << "            "
      << " pT: "  << setw(6) << setprecision(3) << ref->combinedMuon()->pt()
      << " eta: " << setw(6) << setprecision(3) << ref->combinedMuon()->eta()
      << " phi: " << setw(6) << setprecision(3) << ref->combinedMuon()->phi();
  return out;
}

std::ostream& operator<< (std::ostream& out, TrackingParticleRef ref) {
  out << " ["     << setw(4) << ref.index() << "]"
      << " type:" << setw(6) << ref->pdgId() 
      << " pT: "  << setw(6) << setprecision(3) << ref->pt() 
      << " eta: " << setw(6) << setprecision(3) << ref->eta()
      << " phi: " << setw(6) << setprecision(3) << ref->phi();
  return out;
}
 
void printAssociations(const char* label, TrackingParticleRef tp, const reco::SimToRecoCollection& byhits, const reco::SimToRecoCollection& bychi2) {
  reco::SimToRecoCollection::result_type found_byhits;
  if (byhits.find(tp) != byhits.end()) found_byhits = byhits[tp];
  reco::SimToRecoCollection::result_type found_bychi2;
  if (bychi2.find(tp) != bychi2.end()) found_bychi2 = bychi2[tp];

  typedef boost::tuple< std::pair<bool, double>, std::pair<bool, double> > Quality;
  Quality quality;
  std::map<TrackRef, Quality> found;
  for (std::vector<std::pair<TrackRef, double> >::const_iterator it = found_byhits.begin(); it != found_byhits.end(); ++it) {
    const TrackRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<0>().first  = true;
    found[ref].get<0>().second = it->second;
  }
  for (std::vector<std::pair<TrackRef, double> >::const_iterator it = found_bychi2.begin(); it != found_bychi2.end(); ++it) {
    const TrackRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<1>().first  = true;
    found[ref].get<1>().second = it->second;
  }
  
  for (std::map<TrackRef, Quality>::const_iterator it = found.begin(); it != found.end(); ++it) {
    cout << "    " << setw(7) << left << label << it->first;
    if (it->second.get<0>().first) cout << " [" << setw(6) << setprecision(3) << it->second.get<0>().second << "]"; else cout << "         ";
    if (it->second.get<1>().first) cout << " [" << setw(6) << setprecision(3) << it->second.get<1>().second << "]"; else cout << "         ";
    cout << endl;
  }
}

void printAssociations(const char* label, TrackRef tp, const reco::RecoToSimCollection& byhits, const reco::RecoToSimCollection& bychi2) {
  reco::RecoToSimCollection::result_type found_byhits;
  if (byhits.find(tp) != byhits.end()) found_byhits = byhits[tp];
  reco::RecoToSimCollection::result_type found_bychi2;
  if (bychi2.find(tp) != bychi2.end()) found_bychi2 = bychi2[tp];

  typedef boost::tuple< std::pair<bool, double>, std::pair<bool, double> > Quality;
  Quality quality;
  std::map<TrackingParticleRef, Quality> found;
  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = found_byhits.begin(); it != found_byhits.end(); ++it) {
    const TrackingParticleRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<0>().first  = true;
    found[ref].get<0>().second = it->second;
  }
  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = found_bychi2.begin(); it != found_bychi2.end(); ++it) {
    const TrackingParticleRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<1>().first  = true;
    found[ref].get<1>().second = it->second;
  }
  
  for (std::map<TrackingParticleRef, Quality>::const_iterator it = found.begin(); it != found.end(); ++it) {
    cout << "    " << setw(7) << left << label << it->first;
    if (it->second.get<0>().first) cout << " [" << setw(6) << setprecision(3) << it->second.get<0>().second << "]"; else cout << "         ";
    if (it->second.get<1>().first) cout << " [" << setw(6) << setprecision(3) << it->second.get<1>().second << "]"; else cout << "         ";
    cout << endl;
  }
}


template <typename T>
std::ostream& operator<< (std::ostream& out, const std::pair<edm::Ref<T>, double> & assoc) {
  out << assoc.first << " quality: " << assoc.second;
  return out;
}

testLeptonAssociator::testLeptonAssociator(edm::ParameterSet const& iConfig) {
  m_recoTracks      = iConfig.getParameter<edm::InputTag>( "recoTracks" );
  m_standAloneMuons = iConfig.getParameter<edm::InputTag>( "standAloneMuons" );
  m_globalMuons     = iConfig.getParameter<edm::InputTag>( "globalMuons" );
  m_trackingTruth   = iConfig.getParameter<edm::InputTag>( "trackingTruth" );
  m_flavour         = iConfig.getParameter<unsigned int>(  "leptonFlavour" );
  m_ptcut           = iConfig.getParameter<double>(        "minPt" );
}

testLeptonAssociator::~testLeptonAssociator() {
}

void testLeptonAssociator::beginJob(const EventSetup & iSetup) {
  edm::ESHandle<TrackAssociatorBase> associatorByHitsHandle;
  iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", associatorByHitsHandle);
  m_associatorByHits = associatorByHitsHandle.product();

  edm::ESHandle<TrackAssociatorBase> associatorByChi2Handle;
  iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2", associatorByChi2Handle);
  m_associatorByChi2 = associatorByChi2Handle.product();
}

void testLeptonAssociator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  edm::Handle<reco::TrackCollection> recoTrackHandle;
  iEvent.getByLabel(m_recoTracks, recoTrackHandle);
  const reco::TrackCollection& recoTrackCollection = *(recoTrackHandle.product()); 
  
  edm::Handle<reco::TrackCollection> standAloneMuonHandle;
  iEvent.getByLabel(m_standAloneMuons, standAloneMuonHandle);
  const reco::TrackCollection& standAloneMuonCollection = *(standAloneMuonHandle.product()); 
  
  edm::Handle<reco::TrackCollection> globalMuonTrackHandle;
  iEvent.getByLabel(m_globalMuons, globalMuonTrackHandle);
  const reco::TrackCollection& globalMuonTrackCollection = *(globalMuonTrackHandle.product()); 
  
  edm::Handle<reco::MuonCollection> globalMuonHandle;
  iEvent.getByLabel(m_globalMuons, globalMuonHandle);
  const reco::MuonCollection& globalMuonCollection = *(globalMuonHandle.product()); 
  
  edm::Handle<TrackingParticleCollection> trackingParticleHandle ;
  iEvent.getByLabel(m_trackingTruth, trackingParticleHandle);
  const TrackingParticleCollection& trackingParticleCollection = *(trackingParticleHandle.product());

  cout << fixed;

  cout << endl;
  cout << "Found " << setw(6) << trackingParticleCollection.size() << " TrackingParticles" << flush;
  unsigned int count = 0;
  for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i)
    if (
      (abs(trackingParticleCollection[i].pdgId()) == m_flavour) and
      (trackingParticleCollection[i].pt() >= m_ptcut)
    )
      ++count;

  cout << " ( " << setw(2) << count << " leptons with pT above " << m_ptcut << " GeV)" << endl;
  cout << "      " << setw(6) << recoTrackCollection.size()  << " Tracks" << endl;
  cout << "      " << setw(6) << globalMuonCollection.size() << " Leptons" << endl;

  // look for tracks and muons associated to the tracking particles
  {
    reco::SimToRecoCollection byhits_tracks      = m_associatorByHits->associateSimToReco(recoTrackHandle,       trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection byhits_globaltrack = m_associatorByHits->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection byhits_standalone  = m_associatorByHits->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection bychi2_tracks      = m_associatorByChi2->associateSimToReco(recoTrackHandle,       trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection bychi2_globaltrack = m_associatorByChi2->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle, &iEvent );
    reco::SimToRecoCollection bychi2_standalone  = m_associatorByChi2->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle, &iEvent );

    for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i) {
      TrackingParticleRef tp (trackingParticleHandle, i);
      if ((abs(tp->pdgId()) != m_flavour) or (tp->pt() < m_ptcut)) 
        continue;
      cout << "--> TrackingParticle" << tp << endl;
      printAssociations("Track",  tp, byhits_tracks,      bychi2_tracks);
      printAssociations("Local",  tp, byhits_standalone,  bychi2_standalone);
      printAssociations("Global", tp, byhits_globaltrack, bychi2_globaltrack);
    }

  }

  // look for tracking particles associated to the (tracker part of the) reconstructed (global) muons
  reco::RecoToSimCollection byhits_globalfake = m_associatorByHits->associateRecoToSim (recoTrackHandle, trackingParticleHandle, &iEvent );
  reco::RecoToSimCollection bychi2_globalfake = m_associatorByChi2->associateRecoToSim (recoTrackHandle, trackingParticleHandle, &iEvent );
  for (MuonCollection::size_type i = 0; i < globalMuonCollection.size(); ++i) {
    MuonRef lepton(globalMuonHandle, i);
    cout << "<-- Global " << lepton << endl;
    printAssociations("TrackingParticle", lepton->track(), byhits_globalfake, bychi2_globalfake);
  }

  // look for tracking particles associated to the reconstructed (standAlone) muons
  reco::RecoToSimCollection byhits_standalonefake = m_associatorByHits->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle, &iEvent );
  reco::RecoToSimCollection bychi2_standalonefake = m_associatorByChi2->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle, &iEvent );
  for (TrackCollection::size_type i = 0; i < standAloneMuonCollection.size(); ++i) {
    TrackRef lepton(standAloneMuonHandle, i);
    cout << "<-- Local  " << lepton << endl;
    printAssociations("TrackingParticle", lepton, byhits_standalonefake, bychi2_standalonefake);
  }

}

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testLeptonAssociator);
