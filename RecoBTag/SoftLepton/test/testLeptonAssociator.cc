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
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

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
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& setup) override;

private:
  edm::EDGetTokenT<edm::View<reco::Track> >     token_recoTracks;
  edm::EDGetTokenT<edm::View<reco::Track> >     token_standAloneMuons;
  edm::EDGetTokenT<edm::View<reco::Track> >     token_globalMuons;
  edm::EDGetTokenT<reco::MuonCollection>        token_muons;
  edm::EDGetTokenT<TrackingParticleCollection>  token_trackingTruth;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorByHits;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorByChi2;

  unsigned int                                  m_flavour;
  double                                        m_ptcut;
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
  token_recoTracks      = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "tracks" ));
  token_standAloneMuons = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "standAloneMuonTracks" ));
  token_globalMuons     = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>( "globalMuonTracks" ));
  token_muons           = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>( "muons" ));
  token_trackingTruth   = consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>( "trackingTruth" ));
  m_flavour             = iConfig.getParameter<unsigned int>(  "leptonFlavour" );
  m_ptcut               = iConfig.getParameter<double>(        "minPt" );
  token_associatorByHits = consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByHits"));
  token_associatorByChi2 = consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByChi2"));
}

void testLeptonAssociator::analyze(const edm::Event& iEvent, const edm::EventSetup& setup) {

  edm::Handle<edm::View<reco::Track> > recoTrackHandle;
  iEvent.getByToken(token_recoTracks, recoTrackHandle);
  const edm::View<reco::Track> & recoTrackCollection = *(recoTrackHandle.product());

  edm::Handle<edm::View<reco::Track> > standAloneMuonHandle;
  iEvent.getByToken(token_standAloneMuons, standAloneMuonHandle);
  const edm::View<reco::Track>& standAloneMuonCollection = *(standAloneMuonHandle.product());

  edm::Handle<edm::View<reco::Track> > globalMuonTrackHandle;
  iEvent.getByToken(token_globalMuons, globalMuonTrackHandle);
  //const edm::View<reco::Track>& globalMuonTrackCollection = *(globalMuonTrackHandle.product());

  edm::Handle<reco::MuonCollection> globalMuonHandle;
  iEvent.getByToken(token_muons, globalMuonHandle);
  const reco::MuonCollection& globalMuonCollection = *(globalMuonHandle.product());

  edm::Handle<TrackingParticleCollection> trackingParticleHandle ;
  iEvent.getByToken(token_trackingTruth, trackingParticleHandle);
  const TrackingParticleCollection& trackingParticleCollection = *(trackingParticleHandle.product());


  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByHits;
  iEvent.getByToken(token_associatorByHits, associatorByHits);

  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByChi2;
  iEvent.getByToken(token_associatorByChi2, associatorByChi2);

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
    reco::SimToRecoCollection bychi2_tracks      = associatorByChi2->associateSimToReco(recoTrackHandle,       trackingParticleHandle );
    reco::SimToRecoCollection bychi2_globaltrack = associatorByChi2->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle );
    reco::SimToRecoCollection bychi2_standalone  = associatorByChi2->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle );
    reco::SimToRecoCollection byhits_tracks      = associatorByHits->associateSimToReco(recoTrackHandle,       trackingParticleHandle );
    reco::SimToRecoCollection byhits_globaltrack = associatorByHits->associateSimToReco(globalMuonTrackHandle, trackingParticleHandle );
    reco::SimToRecoCollection byhits_standalone  = associatorByHits->associateSimToReco(standAloneMuonHandle,  trackingParticleHandle );

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
  reco::RecoToSimCollection byhits_globalfake = associatorByHits->associateRecoToSim (recoTrackHandle, trackingParticleHandle );
  reco::RecoToSimCollection bychi2_globalfake = associatorByChi2->associateRecoToSim (recoTrackHandle, trackingParticleHandle );
  for (reco::MuonCollection::size_type i = 0; i < globalMuonCollection.size(); ++i) {
    reco::MuonRef lepton(globalMuonHandle, i);
    std::cout << "<-- Global " << lepton << std::endl;
    printAssociations("TrackingParticle", lepton->track(), byhits_globalfake, bychi2_globalfake);
  }

  // look for tracking particles associated to the reconstructed standAlone muons
  reco::RecoToSimCollection byhits_standalonefake = associatorByHits->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle );
  reco::RecoToSimCollection bychi2_standalonefake = associatorByChi2->associateRecoToSim (standAloneMuonHandle, trackingParticleHandle );
  for (edm::View<reco::Track>::size_type i = 0; i < standAloneMuonCollection.size(); ++i) {
    edm::RefToBase<reco::Track> lepton(standAloneMuonHandle, i);
    std::cout << "<-- Local  " << lepton << std::endl;
    printAssociations("TrackingParticle", lepton, byhits_standalonefake, bychi2_standalonefake);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testLeptonAssociator);
