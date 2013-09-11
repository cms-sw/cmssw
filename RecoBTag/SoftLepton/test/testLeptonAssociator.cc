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
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

namespace std {
  template<>
  class less<edm::RefToBase<reco::Track> > {
  public:
    bool operator()(const edm::RefToBase<reco::Track> & x, const edm::RefToBase<reco::Track> & y) const
    {
      return (x.id().processIndex() < y.id().processIndex()) || (x.id().productIndex() < y.id().productIndex()) || (x.key() < y.key()) || false;
    }
  };
}


class testLeptonAssociator : public edm::EDAnalyzer {
public:
  explicit testLeptonAssociator(const edm::ParameterSet& iConfig);
  virtual ~testLeptonAssociator();
  virtual void beginRun(const edm::EventSetup& setup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& setup);

private:
  edm::InputTag m_recoTracks;
  edm::InputTag m_standAloneMuons;
  edm::InputTag m_globalMuons;
  edm::InputTag m_muons;
  edm::InputTag m_trackingTruth;
  unsigned int  m_flavour;
  double        m_ptcut;

  const TrackAssociatorBase* m_associatorByHits;
  const TrackAssociatorBase* m_associatorByChi2;
};


std::ostream& operator<< (std::ostream& out, edm::RefToBase<reco::Track> ref) {
  out << " {"     << std::setw(2) << ref->found() << "}    "
      << " ["     << std::setw(4) << ref.key() << "]"
      << "            "
      << " pT: "  << std::setw(6) << std::setprecision(3) << ref->pt()
      << " eta: " << std::setw(6) << std::setprecision(3) << ref->eta()
      << " phi: " << std::setw(6) << std::setprecision(3) << ref->phi();
  return out;
}

std::ostream& operator<< (std::ostream& out, reco::MuonRef ref) {
  if (ref->isGlobalMuon()) {
    out << " {"     << std::setw(2) << ref->track()->found() << "+" << std::setw(2) << ref->standAloneMuon()->found() << "} " 
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->globalTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->globalTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->globalTrack()->phi();
  } else if (ref->isTrackerMuon()) {
    out << " {"     << std::setw(2) << ref->track()->found() << "   } " 
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->innerTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->innerTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->innerTrack()->phi();
  } else if (ref->isStandAloneMuon()) {
    out << " {   "  << std::setw(2) << ref->standAloneMuon()->found() << "} " 
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->outerTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->outerTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->outerTrack()->phi();
  } else {
    out << "(mun track not available)";
  }
  return out;
}

std::ostream& operator<< (std::ostream& out, TrackingParticleRef ref) {
  out << " ["     << std::setw(4) << ref.key() << "]"
      << " type:" << std::setw(6) << ref->pdgId() 
      << " pT: "  << std::setw(6) << std::setprecision(3) << ref->pt() 
      << " eta: " << std::setw(6) << std::setprecision(3) << ref->eta()
      << " phi: " << std::setw(6) << std::setprecision(3) << ref->phi();
  return out;
}
 
void printAssociations(const char* label, TrackingParticleRef tp, const reco::SimToRecoCollection& byhits, const reco::SimToRecoCollection& bychi2) {
  reco::SimToRecoCollection::result_type found_byhits;
  if (byhits.find(tp) != byhits.end()) found_byhits = byhits[tp];
  reco::SimToRecoCollection::result_type found_bychi2;
  if (bychi2.find(tp) != bychi2.end()) found_bychi2 = bychi2[tp];

  typedef boost::tuple<double, double> Quality;
  Quality quality;
  std::map<edm::RefToBase<reco::Track>, Quality> found;
  for (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator it = found_byhits.begin(); it != found_byhits.end(); ++it) {
    const edm::RefToBase<reco::Track> ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<0>() = it->second;
  }
  for (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator it = found_bychi2.begin(); it != found_bychi2.end(); ++it) {
    const edm::RefToBase<reco::Track> ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<1>() = - it->second;  // why is chi2 negative ?
  }
  
  for (std::map<edm::RefToBase<reco::Track>, Quality>::const_iterator it = found.begin(); it != found.end(); ++it) {
    std::cout << "    " << std::setw(7) << std::left << label << std::right << it->first;
    if (it->second.get<0>()) std::cout << " [" << std::setw(6) << std::setprecision(3) << it->second.get<0>() << "]"; else std::cout << "         ";
    if (it->second.get<1>()) std::cout << " [" << std::setw(6) << std::setprecision(3) << it->second.get<1>() << "]"; else std::cout << "         ";
    std::cout << std::endl;
  }
}

void printAssociations(const char* label, edm::RefToBase<reco::Track> tp, const reco::RecoToSimCollection& byhits, const reco::RecoToSimCollection& bychi2) {
  reco::RecoToSimCollection::result_type found_byhits;
  if (byhits.find(tp) != byhits.end()) found_byhits = byhits[tp];
  reco::RecoToSimCollection::result_type found_bychi2;
  if (bychi2.find(tp) != bychi2.end()) found_bychi2 = bychi2[tp];

  typedef boost::tuple<double, double> Quality;
  Quality quality;
  std::map<TrackingParticleRef, Quality> found;
  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = found_byhits.begin(); it != found_byhits.end(); ++it) {
    const TrackingParticleRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<0>() = it->second;
  }
  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = found_bychi2.begin(); it != found_bychi2.end(); ++it) {
    const TrackingParticleRef ref = it->first;
    found.insert( std::make_pair(ref, Quality()) );
    found[ref].get<1>() = - it->second;  // why is chi2 negative ?
  }
  
  for (std::map<TrackingParticleRef, Quality>::const_iterator it = found.begin(); it != found.end(); ++it) {
    std::cout << "    " << std::setw(7) << std::left << label << std::right << it->first;
    if (it->second.get<0>()) std::cout << " [" << std::setw(6) << std::setprecision(3) << it->second.get<0>() << "]"; else std::cout << "         ";
    if (it->second.get<1>()) std::cout << " [" << std::setw(6) << std::setprecision(3) << it->second.get<1>() << "]"; else std::cout << "         ";
    std::cout << std::endl;
  }
}

void printAssociations(const char* label, reco::TrackRef tp, const reco::RecoToSimCollection& byhits, const reco::RecoToSimCollection& bychi2) {
  printAssociations(label, edm::RefToBase<reco::Track>(tp), byhits, bychi2);
}
  

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::pair<edm::Ref<T>, double> & assoc) {
  out << assoc.first << " quality: " << assoc.second;
  return out;
}

testLeptonAssociator::testLeptonAssociator(edm::ParameterSet const& iConfig) {
  m_recoTracks      = iConfig.getParameter<edm::InputTag>( "tracks" );
  m_standAloneMuons = iConfig.getParameter<edm::InputTag>( "standAloneMuonTracks" );
  m_globalMuons     = iConfig.getParameter<edm::InputTag>( "globalMuonTracks" );
  m_muons           = iConfig.getParameter<edm::InputTag>( "muons" );
  m_trackingTruth   = iConfig.getParameter<edm::InputTag>( "trackingTruth" );
  m_flavour         = iConfig.getParameter<unsigned int>(  "leptonFlavour" );
  m_ptcut           = iConfig.getParameter<double>(        "minPt" );
}

testLeptonAssociator::~testLeptonAssociator() {
}

void testLeptonAssociator::beginRun(const edm::EventSetup & setup) {
  edm::ESHandle<TrackAssociatorBase> associatorByHitsHandle;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", associatorByHitsHandle);
  m_associatorByHits = associatorByHitsHandle.product();

  edm::ESHandle<TrackAssociatorBase> associatorByChi2Handle;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2", associatorByChi2Handle);
  m_associatorByChi2 = associatorByChi2Handle.product();
}

void testLeptonAssociator::analyze(const edm::Event& iEvent, const edm::EventSetup& setup) {
  
  edm::Handle<edm::View<reco::Track> > recoTrackHandle;
  iEvent.getByLabel(m_recoTracks, recoTrackHandle);
  const edm::View<reco::Track> & recoTrackCollection = *(recoTrackHandle.product()); 
  
  edm::Handle<edm::View<reco::Track> > standAloneMuonHandle;
  iEvent.getByLabel(m_standAloneMuons, standAloneMuonHandle);
  const edm::View<reco::Track>& standAloneMuonCollection = *(standAloneMuonHandle.product()); 
  
  edm::Handle<edm::View<reco::Track> > globalMuonTrackHandle;
  iEvent.getByLabel(m_globalMuons, globalMuonTrackHandle);
  //const edm::View<reco::Track>& globalMuonTrackCollection = *(globalMuonTrackHandle.product()); 
  
  edm::Handle<reco::MuonCollection> globalMuonHandle;
  iEvent.getByLabel(m_muons, globalMuonHandle);
  const reco::MuonCollection& globalMuonCollection = *(globalMuonHandle.product()); 
  
  edm::Handle<TrackingParticleCollection> trackingParticleHandle ;
  iEvent.getByLabel(m_trackingTruth, trackingParticleHandle);
  const TrackingParticleCollection& trackingParticleCollection = *(trackingParticleHandle.product());

  std::cout << std::fixed;

  std::cout << std::endl;
  std::cout << "Found " << std::setw(6) << trackingParticleCollection.size() << " TrackingParticles" << std::flush;
  unsigned int count = 0;
  for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i)
    if (
      (std::abs(trackingParticleCollection[i].pdgId()) == (int)m_flavour) and
      (trackingParticleCollection[i].pt() >= m_ptcut)
    )
      ++count;

  std::cout << " ( " << std::setw(2) << count << " leptons with pT above " << m_ptcut << " GeV)"  << std::endl;
  std::cout << "      " << std::setw(6) << recoTrackCollection.size()      << " Tracks"           << std::endl;
  std::cout << "      " << std::setw(6) << globalMuonCollection.size()     << " Global muons"     << std::endl;
  std::cout << "      " << std::setw(6) << standAloneMuonCollection.size() << " StandAlone muons" << std::endl;

  // look for tracks and muons associated to the tracking particles
  {
    reco::SimToRecoCollection bychi2_tracks      = m_associatorByChi2->associateSimToReco(recoTrackHandle,       trackingParticleHandle, &iEvent, &setup );
    reco::SimToRecoCollection bychi2_globaltrack = m_associatorByChi2->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle, &iEvent, &setup );
    reco::SimToRecoCollection bychi2_standalone  = m_associatorByChi2->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle, &iEvent, &setup );
    reco::SimToRecoCollection byhits_tracks      = m_associatorByHits->associateSimToReco(recoTrackHandle,       trackingParticleHandle, &iEvent, &setup );
    reco::SimToRecoCollection byhits_globaltrack = m_associatorByHits->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle, &iEvent, &setup );
    reco::SimToRecoCollection byhits_standalone  = m_associatorByHits->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle, &iEvent, &setup );

    for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i) {
      TrackingParticleRef tp (trackingParticleHandle, i);
      if ((std::abs(tp->pdgId()) != (int)m_flavour) or (tp->pt() < m_ptcut)) 
        continue;
      std::cout << "--> TrackingParticle" << tp << std::endl;
      printAssociations("Track",  tp, byhits_tracks,      bychi2_tracks);
      printAssociations("Local",  tp, byhits_standalone,  bychi2_standalone);
      printAssociations("Global", tp, byhits_globaltrack, bychi2_globaltrack);
    }
  }
  
  // look for tracking particles associated to the (tracker part of the) reconstructed global muons
  reco::RecoToSimCollection byhits_globalfake = m_associatorByHits->associateRecoToSim (recoTrackHandle, trackingParticleHandle, &iEvent, &setup );
  reco::RecoToSimCollection bychi2_globalfake = m_associatorByChi2->associateRecoToSim (recoTrackHandle, trackingParticleHandle, &iEvent, &setup );
  for (reco::MuonCollection::size_type i = 0; i < globalMuonCollection.size(); ++i) {
    reco::MuonRef lepton(globalMuonHandle, i);
    std::cout << "<-- Global " << lepton << std::endl;
    printAssociations("TrackingParticle", lepton->track(), byhits_globalfake, bychi2_globalfake);
  }
  
  // look for tracking particles associated to the reconstructed standAlone muons
  reco::RecoToSimCollection byhits_standalonefake = m_associatorByHits->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle, &iEvent, &setup );
  reco::RecoToSimCollection bychi2_standalonefake = m_associatorByChi2->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle, &iEvent, &setup );
  for (edm::View<reco::Track>::size_type i = 0; i < standAloneMuonCollection.size(); ++i) {
    edm::RefToBase<reco::Track> lepton(standAloneMuonHandle, i);
    std::cout << "<-- Local  " << lepton << std::endl;
    printAssociations("TrackingParticle", lepton, byhits_standalonefake, bychi2_standalonefake);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testLeptonAssociator);
