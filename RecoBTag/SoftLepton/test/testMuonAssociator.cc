#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <set>

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
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

namespace std {
  template<>
  class less<edm::RefToBase<reco::Track> > {
  public:
    bool operator()(const edm::RefToBase<reco::Track> & x, const edm::RefToBase<reco::Track> & y) const
    {
      return (x.id().id() < y.id().id()) || (x.key() < y.key()) || false;
    }
  };
}


class testMuonAssociator : public edm::EDAnalyzer {
public:
  explicit testMuonAssociator(const edm::ParameterSet& iConfig);
  virtual ~testMuonAssociator();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

private:
  edm::EDGetTokenT<edm::View<reco::Track> >     token_recoTracks;
  edm::EDGetTokenT<edm::View<reco::Track> >     token_standAloneMuons;
  edm::EDGetTokenT<edm::View<reco::Track> >     token_globalMuons;
  edm::EDGetTokenT<reco::MuonCollection>        token_muons;
  edm::EDGetTokenT<TrackingParticleCollection>  token_trackingTruth;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorByHits;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorByChi2;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorByPos;
  unsigned int                                  m_flavour;
  double                                        m_ptcut;
};


std::ostream& operator<< (std::ostream& out, edm::RefToBase<reco::Track> ref) {
  out << std::fixed
      << " {"     << std::setw(2) << ref->found() << "}    "
      << " ["     << std::setw(4) << ref.key() << "]"
      << "            "
      << " pT: "  << std::setw(6) << std::setprecision(3) << ref->pt()
      << " eta: " << std::setw(6) << std::setprecision(3) << ref->eta()
      << " phi: " << std::setw(6) << std::setprecision(3) << ref->phi();
  return out;
}

std::ostream& operator<< (std::ostream& out, reco::MuonRef ref) {
  out << std::fixed;
  if (ref->isGlobalMuon()) {
    out << " {"     << std::setw(2) << ref->innerTrack()->found() << "+" << std::setw(2) << ref->outerTrack()->found() << "} "
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->globalTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->globalTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->globalTrack()->phi();
  } else if (ref->isTrackerMuon()) {
    out << " {"     << std::setw(2) << ref->innerTrack()->found() << "   } "
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->innerTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->innerTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->innerTrack()->phi();
  } else if (ref->isStandAloneMuon()) {
    out << " {   "  << std::setw(2) << ref->outerTrack()->found() << "} "
        << " ["     << std::setw(4) << ref.key() << "]"
        << "            "
        << " pT: "  << std::setw(6) << std::setprecision(3) << ref->outerTrack()->pt()
        << " eta: " << std::setw(6) << std::setprecision(3) << ref->outerTrack()->eta()
        << " phi: " << std::setw(6) << std::setprecision(3) << ref->outerTrack()->phi();
  } else {
    out << "(muon track not available)";
  }
  return out;
}

std::ostream& operator<< (std::ostream& out, TrackingParticleRef ref) {
  out << std::fixed;
  out << " ["     << std::setw(4) << ref.key() << "]"
      << " type:" << std::setw(6) << ref->pdgId()
      << " pT: "  << std::setw(6) << std::setprecision(3) << ref->pt()
      << " eta: " << std::setw(6) << std::setprecision(3) << ref->eta()
      << " phi: " << std::setw(6) << std::setprecision(3) << ref->phi();
  return out;
}

struct Quality {
  double byChi2;
  double byHits;
  double byPosition;
};

template <class Ref>
class Associations {
public:
  typedef std::map<Ref, Quality>                map_type;
  typedef std::vector<std::pair<Ref, double> >  association_type;

  template <class Key, class Associator>
  Associations(const std::string & label, const Key & candidate, const Associator & byHits, const Associator & byChi2, const Associator & byPosition) :
    m_map(),
    m_label(label)
  {
    if (byHits.find(candidate) != byHits.end())
      fillByHits( byHits[candidate] );
    if (byChi2.find(candidate) != byChi2.end())
      fillByChi2( byChi2[candidate] );
    if (byPosition.find(candidate) != byPosition.end())
      fillByPosition( byPosition[candidate] );
  }

  void fillByHits(const association_type & found) {
    for (typename association_type::const_iterator it = found.begin(); it != found.end(); ++it) {
      const Ref & ref = it->first;
      m_map.insert( std::make_pair(ref, Quality()) );
      m_map[ref].byHits = it->second;
    }
  }

  void fillByChi2(const association_type & found) {
    for (typename association_type::const_iterator it = found.begin(); it != found.end(); ++it) {
      const Ref & ref = it->first;
      m_map.insert( std::make_pair(ref, Quality()) );
      m_map[ref].byChi2 = - it->second;
    }
  }

  void fillByPosition(const association_type & found) {
    for (typename association_type::const_iterator it = found.begin(); it != found.end(); ++it) {
      const Ref & ref = it->first;
      m_map.insert( std::make_pair(ref, Quality()) );
      m_map[ref].byPosition = - it->second;
    }
  }

  const map_type & map(void) const {
    return m_map;
  }

  void dump(std::ostream & out) const {
    std::stringstream buffer;
    for (typename map_type::const_iterator it = m_map.begin(); it != m_map.end(); ++it) {
      buffer << "    " << std::setw(7) << std::left << m_label << std::right << it->first;
      if (it->second.byHits)
        buffer << " [" << std::setw(6) << std::setprecision(3) << it->second.byHits << "]";
      else
        buffer << "         ";
      if (it->second.byChi2)
        buffer << " [" << std::setw(6) << std::setprecision(3) << it->second.byChi2 << "]";
      else
        buffer << "         ";
      if (it->second.byPosition)
        buffer << " [" << std::setw(6) << std::setprecision(3) << it->second.byPosition << "]";
      else
        buffer << "         ";
      buffer << std::endl;
    }
    out << buffer.str();
  }

private:
  map_type    m_map;
  std::string m_label;

};

template <class Ref>
std::ostream & operator<<(std::ostream & out, const Associations<Ref> & association) {
  association.dump(out);
  return out;
}

template <typename T>
std::ostream & operator<< (std::ostream& out, const std::pair<edm::Ref<T>, double> & assoc) {
  out << assoc.first << " quality: " << assoc.second;
  return out;
}

testMuonAssociator::testMuonAssociator(edm::ParameterSet const& iConfig) {
  token_recoTracks      = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "tracks" ));
  token_standAloneMuons = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "standAloneMuonTracks" ));
  token_globalMuons     = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "globalMuonTracks" ));
  token_muons           = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>( "muons" ));
  token_trackingTruth   = consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>( "trackingTruth" ));
  token_associatorByHits= consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByHits"));
  token_associatorByChi2= consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByChi2"));
  token_associatorByPos = consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByPosition"));
  m_flavour             = iConfig.getParameter<unsigned int>(  "leptonFlavour" );
  m_ptcut               = iConfig.getParameter<double>(        "minPt" );
}

testMuonAssociator::~testMuonAssociator() {
}

void testMuonAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // access EventSetup during the Event loop, to make sure conditions are up to date
  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByHits;
  event.getByToken(token_associatorByHits, associatorByHits);

  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByChi2;
  event.getByToken(token_associatorByChi2, associatorByChi2);

  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByPos;
  event.getByToken(token_associatorByPos, associatorByPos);

  // access Event collections
  edm::Handle<edm::View<reco::Track> > recoTrackHandle;
  event.getByToken(token_recoTracks, recoTrackHandle);
  const edm::View<reco::Track> & recoTrackCollection = *(recoTrackHandle.product());

  edm::Handle<edm::View<reco::Track> > standAloneMuonHandle;
  event.getByToken(token_standAloneMuons, standAloneMuonHandle);
  const edm::View<reco::Track>& standAloneMuonCollection = *(standAloneMuonHandle.product());

  edm::Handle<edm::View<reco::Track> > globalMuonTrackHandle;
  event.getByToken(token_globalMuons, globalMuonTrackHandle);
  const edm::View<reco::Track>& globalMuonTrackCollection = *(globalMuonTrackHandle.product());

  edm::Handle<reco::MuonCollection> muonHandle;
  event.getByToken(token_muons, muonHandle);
  const reco::MuonCollection& muonCollection = *(muonHandle.product());

  edm::Handle<TrackingParticleCollection> trackingParticleHandle ;
  event.getByToken(token_trackingTruth, trackingParticleHandle);
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
  std::cout << "      " << std::setw(6) << muonCollection.size()     << " Global muons"     << std::endl;
  std::cout << "      " << std::setw(6) << standAloneMuonCollection.size() << " StandAlone muons" << std::endl;
  std::cout << std::endl;

  // look for tracks and muons associated to the tracking particles
  {
    reco::SimToRecoCollection bychi2_tracks;
    reco::SimToRecoCollection bychi2_globaltrack;
    reco::SimToRecoCollection bychi2_standalone;
    reco::SimToRecoCollection bypos_tracks;
    reco::SimToRecoCollection bypos_globaltrack;
    reco::SimToRecoCollection bypos_standalone;
    reco::SimToRecoCollection byhits_tracks;
    reco::SimToRecoCollection byhits_globaltrack;
    reco::SimToRecoCollection byhits_standalone;
    try {
      bychi2_tracks      = associatorByChi2->associateSimToReco(recoTrackHandle,       trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      bychi2_globaltrack = associatorByChi2->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      bychi2_standalone  = associatorByChi2->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      byhits_tracks      = associatorByHits->associateSimToReco(recoTrackHandle,       trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      byhits_globaltrack = associatorByHits->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      byhits_standalone  = associatorByHits->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    /*
    try {
      bypos_tracks       = associatorByPos ->associateSimToReco(recoTrackHandle,       trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      bypos_globaltrack  = associatorByPos ->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    try {
      bypos_standalone   = associatorByPos ->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle );
    } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
    */

    for (TrackingParticleCollection::size_type i = 0; i < trackingParticleCollection.size(); ++i) {
      TrackingParticleRef tp (trackingParticleHandle, i);
      if ((std::abs(tp->pdgId()) != (int)m_flavour) or (tp->pt() < m_ptcut))
        continue;
      std::cout << "--> TrackingParticle" << tp << std::endl;
      std::cout << Associations<edm::RefToBase<reco::Track> >("Track",  tp, byhits_tracks,      bychi2_tracks,      bypos_tracks);
      std::cout << Associations<edm::RefToBase<reco::Track> >("Local",  tp, byhits_standalone,  bychi2_standalone,  bypos_standalone);
      std::cout << Associations<edm::RefToBase<reco::Track> >("Global", tp, byhits_globaltrack, bychi2_globaltrack, bypos_globaltrack);
    }
  }

  // look for tracking particles associated to the (tracker part of the) reconstructed global muons
  reco::RecoToSimCollection byhits_muon;
  reco::RecoToSimCollection bychi2_muon;
  reco::RecoToSimCollection bypos_muon;
  try {
    byhits_muon = associatorByHits->associateRecoToSim (recoTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
  try {
    bychi2_muon = associatorByChi2->associateRecoToSim (recoTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
  /*
  try {
    bypos_muon  = associatorByPos ->associateRecoToSim (recoTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::SimToRecoCollection:\n" << e.what() << std::endl; }
  */
  for (reco::MuonCollection::size_type i = 0; i < muonCollection.size(); ++i) {
    reco::MuonRef muon(muonHandle, i);
    std::cout << "<-- Muon   " << muon << std::endl;
    std::cout << Associations<TrackingParticleRef>("TrackingParticle", edm::RefToBase<reco::Track>(muon->track()), byhits_muon, bychi2_muon, bypos_muon);
  }

  // look for tracking particles associated to the reconstructed Global muons
  reco::RecoToSimCollection byhits_global;
  reco::RecoToSimCollection bychi2_global;
  reco::RecoToSimCollection bypos_global;
  try {
    byhits_global = associatorByHits->associateRecoToSim (globalMuonTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  try {
    bychi2_global = associatorByChi2->associateRecoToSim (globalMuonTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  /*
  try {
    bypos_global  = associatorByPos ->associateRecoToSim (globalMuonTrackHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  */
  for (edm::View<reco::Track>::size_type i = 0; i < globalMuonTrackCollection.size(); ++i) {
    edm::RefToBase<reco::Track> track(globalMuonTrackHandle, i);
    std::cout << "<-- Local  " << track << std::endl;
    std::cout << Associations<TrackingParticleRef>("TrackingParticle", track, byhits_global, bychi2_global, bypos_global);
  }

  // look for tracking particles associated to the reconstructed StandAlone muons
  reco::RecoToSimCollection byhits_standalone;
  reco::RecoToSimCollection bychi2_standalone;
  reco::RecoToSimCollection bypos_standalone;
  try {
    byhits_standalone = associatorByHits->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  try {
    bychi2_standalone = associatorByChi2->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  /*
  try {
    bypos_standalone  = associatorByPos ->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle );
  } catch (const std::exception & e) { std::cerr << std::endl << "Error building reco::RecoToSimCollection:\n" << e.what() << std::endl; }
  */
  for (edm::View<reco::Track>::size_type i = 0; i < standAloneMuonCollection.size(); ++i) {
    edm::RefToBase<reco::Track> track(standAloneMuonHandle, i);
    std::cout << "<-- Local  " << track << std::endl;
    std::cout << Associations<TrackingParticleRef>("TrackingParticle", track, byhits_standalone, bychi2_standalone, bypos_standalone);
  }

  std::cout << std::endl;

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testMuonAssociator);
