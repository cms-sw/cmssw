#include <set>

#include "RecoJets/JetProducers/interface/JetMatchingTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;

namespace {

  template <class T>
  const CaloRecHit* getHit (const T& fCollection, DetId fId) {
    typename T::const_iterator hit = fCollection.find (fId);
    if (hit != fCollection.end()) return &*hit;
    return 0;
  }

  std::vector <const PCaloHit*> getSimHits (const PCaloHitContainer& fCollection, DetId fId) {
    std::vector <const PCaloHit*> result;
    for (unsigned i = 0; i < fCollection.size (); ++i) {
      if (fCollection[i].id() == fId.rawId()) {
	result.push_back (&(fCollection[i]));
      }
    }
    return result;
  }
}

JetMatchingTools::JetMatchingTools (const edm::Event& fEvent) 
  : mEvent (&fEvent),
    mEBRecHitCollection (0),
    mEERecHitCollection (0),
    mHBHERecHitCollection (0),
    mHORecHitCollection (0),
    mHFRecHitCollection (0),
    mEBSimHitCollection (0),
    mEESimHitCollection (0),
    mHcalSimHitCollection (0),
    mSimTrackCollection (0),
    mSimVertexCollection (0),
    mGenParticleCollection (0)
{}

JetMatchingTools::~JetMatchingTools () {}

const EBRecHitCollection* JetMatchingTools::getEBRecHitCollection () {
  if (!mEBRecHitCollection) {
    edm::Handle<EBRecHitCollection> recHits;
    mEvent->getByLabel (edm::InputTag ("ecalRecHit:EcalRecHitsEB"), recHits);
    mEBRecHitCollection = &*recHits;
  }
  return mEBRecHitCollection;
}
const EERecHitCollection* JetMatchingTools::getEERecHitCollection () {
  if (!mEERecHitCollection) {
    edm::Handle<EERecHitCollection> recHits;
    mEvent->getByLabel (edm::InputTag ("ecalRecHit:EcalRecHitsEE"), recHits);
    mEERecHitCollection = &*recHits;
  }
  return mEERecHitCollection;
}
const HBHERecHitCollection* JetMatchingTools::getHBHERecHitCollection () {
  if (!mHBHERecHitCollection) {
    edm::Handle<HBHERecHitCollection> recHits;
    mEvent->getByLabel (edm::InputTag ("hbhereco"), recHits);
    mHBHERecHitCollection = &*recHits;
  }
  return mHBHERecHitCollection;
}
const HORecHitCollection* JetMatchingTools::getHORecHitCollection () {
  if (!mHORecHitCollection) {
    edm::Handle<HORecHitCollection> recHits;
    mEvent->getByLabel (edm::InputTag ("horeco"), recHits);
    mHORecHitCollection = &*recHits;
  }
  return mHORecHitCollection;
}
const HFRecHitCollection* JetMatchingTools::getHFRecHitCollection () {
  if (!mHFRecHitCollection) {
    edm::Handle<HFRecHitCollection> recHits;
    mEvent->getByLabel (edm::InputTag ("hfreco"), recHits);
    mHFRecHitCollection = &*recHits;
  }
  return mHFRecHitCollection;
}
const PCaloHitContainer* JetMatchingTools::getEBSimHitCollection () {
  if (!mEBSimHitCollection) {
    edm::Handle<PCaloHitContainer> simHits;
    mEvent->getByLabel (edm::InputTag ("g4SimHits:EcalHitsEB"), simHits);
    mEBSimHitCollection = &*simHits;
  }
  return mEBSimHitCollection;
}
const PCaloHitContainer* JetMatchingTools::getEESimHitCollection () {
  if (!mEESimHitCollection) {
    edm::Handle<PCaloHitContainer> simHits;
    mEvent->getByLabel (edm::InputTag ("g4SimHits:EcalHitsEE"), simHits);
    mEESimHitCollection = &*simHits;
  }
  return mEESimHitCollection;
}
const PCaloHitContainer* JetMatchingTools::getHcalSimHitCollection () {
  if (!mHcalSimHitCollection) {
    edm::Handle<PCaloHitContainer> simHits;
    mEvent->getByLabel (edm::InputTag ("g4SimHits:HcalHits"), simHits);
    mHcalSimHitCollection = &*simHits;
  }
  return mHcalSimHitCollection;
}
const SimTrackContainer* JetMatchingTools::getSimTrackCollection () {
  if (!mSimTrackCollection) {
    edm::Handle<SimTrackContainer> simHits;
    mEvent->getByLabel (edm::InputTag ("g4SimHits"), simHits);
    mSimTrackCollection = &*simHits;
  }
  return mSimTrackCollection;
}
const SimVertexContainer* JetMatchingTools::getSimVertexCollection () {
  if (!mSimVertexCollection) {
    edm::Handle<SimVertexContainer> simHits;
    mEvent->getByLabel (edm::InputTag ("g4SimHits"), simHits);
    mSimVertexCollection = &*simHits;
  }
  return mSimVertexCollection;
}
const reco::CandidateCollection* JetMatchingTools::getGenParticlesCollection () {
  if (!mGenParticleCollection) {
    edm::Handle<reco::CandidateCollection> handle;
    mEvent->getByLabel (edm::InputTag ("genParticleCandidates"), handle);
    mGenParticleCollection = &*handle;
  }
  return mGenParticleCollection;
}

  /// get towers contributing to CaloJet
std::vector <const CaloTower*> JetMatchingTools::getConstituents (const reco::CaloJet& fJet ) {
  std::vector <const CaloTower*> result;
  std::vector<CaloTowerPtr> constituents = fJet.getCaloConstituents ();
  for (unsigned i = 0; i < constituents.size(); ++i) result.push_back (&*(constituents[i]));
  return result;
}
/// get CaloRecHits contributing to the tower
std::vector <const CaloRecHit*> JetMatchingTools::getConstituents (const CaloTower& fTower) {
  std::vector <const CaloRecHit*> result;
  for (unsigned i = 0; i < fTower.constituentsSize(); ++i) {
    DetId id = fTower.constituent (i);
    const CaloRecHit* hit = 0;
    if (id.det () == DetId::Ecal) {
      if ((EcalSubdetector) id.subdetId () == EcalBarrel) {
 	hit = getHit (*getEBRecHitCollection (), id);
      } 
      else if ((EcalSubdetector) id.subdetId () == EcalEndcap) {
	hit = getHit (*getEERecHitCollection (), id);
      }
    }
    else if (id.det () == DetId::Hcal) {
      if ((HcalSubdetector) id.subdetId () == HcalBarrel || (HcalSubdetector) id.subdetId () == HcalEndcap) {
	hit = getHit (*getHBHERecHitCollection (), id);
      }
      else if ((HcalSubdetector) id.subdetId () == HcalOuter) {
	hit = getHit (*getHORecHitCollection (), id);
      }
      if ((HcalSubdetector) id.subdetId () == HcalForward) {
	hit = getHit (*getHFRecHitCollection (), id);
      }
    }
    if (hit) result.push_back (hit);
    else std::cerr << "Can not find rechit for id " << id.rawId () << std::endl;
  } 
  return result;
}

  /// get cells contributing to the tower
std::vector <DetId> JetMatchingTools::getConstituentIds (const CaloTower& fTower) {
  std::vector <DetId> result;
  for (unsigned i = 0; i < fTower.constituentsSize(); ++i) {
    DetId id = fTower.constituent (i);
    result.push_back (id);
  }
  return result;
}
/// get PCaloHits contributing to the detId
std::vector <const PCaloHit*> JetMatchingTools::getPCaloHits (DetId fId) {
  std::vector <const PCaloHit*> result;
  if (fId.det () == DetId::Ecal) {
    if ((EcalSubdetector) fId.subdetId () == EcalBarrel) {
      result = getSimHits (*getEBSimHitCollection (), fId);
    } 
    else if ((EcalSubdetector) fId.subdetId () == EcalEndcap) {
      result = getSimHits (*getEESimHitCollection (), fId);
    }
  }
  else if (fId.det () == DetId::Hcal) {
    result = getSimHits (*getHcalSimHitCollection (), fId);
  }
  return result;
}
  /// GEANT track ID
int JetMatchingTools::getTrackId (const PCaloHit& fHit) {
  return fHit.geantTrackId ();
}
/// convert trackId to SimTrack
const SimTrack* JetMatchingTools::getTrack (unsigned fSimTrackId) {
  for (unsigned i = 0; i < getSimTrackCollection ()->size (); ++i) {
    if ((*getSimTrackCollection ())[i].trackId() == fSimTrackId) return &(*getSimTrackCollection ())[i];
  }
  return 0;
}
  /// Generator ID
int JetMatchingTools::generatorId (unsigned fSimTrackId) {
  const SimTrack* track = getTrack (fSimTrackId);
  if (!track) return -1;
  while (track->noGenpart ()) {
    if (track->noVertex ()) {
      std::cerr << "JetMatchingTools::generatorId-> No vertex for track " << *track << std::endl;
      return -1;
    }
    const SimVertex* vertex = &((*getSimVertexCollection ())[track->vertIndex ()]);
    if (vertex->noParent()) {
      std::cerr << "JetMatchingTools::generatorId-> No track for vertex " << *vertex << std::endl;
      return -1;
    }
    track = getTrack (vertex->parentIndex ());
  }
  return track->genpartIndex ();
}

  /// GenParticle
const reco::GenParticle* JetMatchingTools::getGenParticle (int fGeneratorId) {
  if (fGeneratorId > int (getGenParticlesCollection ()->size())) {
    std::cerr << "JetMatchingTools::getGenParticle-> requested index " << fGeneratorId << " is grater then container size " << getGenParticlesCollection ()->size() << std::endl;
    return 0;
  }
  return reco::GenJet::genParticle ( &(*getGenParticlesCollection ())[fGeneratorId-1]); // knowhow: index is shifted by 1
}

/// GenParticles for CaloJet
std::vector <const reco::GenParticle*> JetMatchingTools::getGenParticles (const reco::CaloJet& fJet, bool fVerbose) {
  std::set <const reco::GenParticle*> result;
  // follow the chain
  std::vector <const CaloTower*> towers = getConstituents (fJet) ;
  for (unsigned itower = 0; itower < towers.size (); ++itower) {
    std::vector <DetId> detids = getConstituentIds (*(towers[itower])) ;
    for (unsigned iid = 0; iid < detids.size(); ++iid) {
      std::vector <const PCaloHit*> phits = getPCaloHits (detids[iid]);
      for (unsigned iphit = 0; iphit < phits.size(); ++iphit) {
	int trackId = getTrackId (*(phits[iphit]));
	if (trackId >= 0) {
	  int genId = generatorId (trackId);
	  if (genId >= 0) {
	    const reco::GenParticle* genPart = getGenParticle (genId);
	    if (genPart) {
	      result.insert (genPart);
	    }
	    else if (fVerbose) {
	      std::cerr << "JetMatchingTools::getGenParticles-> Can not convert genId " << genId << " to GenParticle" << std::endl;
	    }
	  }
	  else if (fVerbose) {
	    std::cerr << "JetMatchingTools::getGenParticles-> Can not convert trackId " << trackId << " to genId" << std::endl;
	  }
	}
	else if (fVerbose) {
	  std::cerr << "JetMatchingTools::getGenParticles-> Unknown trackId for PCaloHit " << *(phits[iphit]) << std::endl;
	}
      }
    }
  }
  return std::vector <const reco::GenParticle*> (result.begin (), result.end());
}

/// GenParticles for GenJet
std::vector <const reco::GenParticle*> JetMatchingTools::getGenParticles (const reco::GenJet& fJet) {
  return fJet.getGenConstituents ();
}

  /// energy in broken links
double JetMatchingTools::lostEnergyFraction (const reco::CaloJet& fJet ) {
  double totalEnergy = 0;
  double lostEnergy = 0;
  // follow the chain
  std::vector <const CaloTower*> towers = getConstituents (fJet) ;
  for (unsigned itower = 0; itower < towers.size (); ++itower) {
    std::vector <const CaloRecHit*> recHits = getConstituents (*(towers[itower]));
    for (unsigned ihit = 0; ihit < recHits.size(); ++ihit) {
      double foundSimEnergy = 0;
      double lostSimEnergy = 0;
      std::vector <const PCaloHit*> phits = getPCaloHits (recHits[ihit]->detid());
      for (unsigned iphit = 0; iphit < phits.size(); ++iphit) {
	double simEnergy = phits[iphit]->energy ();
	int trackId = getTrackId (*(phits[iphit]));
	if (trackId < 0 || generatorId (trackId) < 0)   lostSimEnergy += simEnergy;
	else   foundSimEnergy += simEnergy;
      }
      if (foundSimEnergy > 0 || lostSimEnergy > 0) {
	totalEnergy += recHits[ihit]->energy ();
	lostEnergy += recHits[ihit]->energy () * lostSimEnergy / (foundSimEnergy + lostSimEnergy);
      }
    }
  }
  return lostEnergy / totalEnergy;
}

  /// energy overlap
double JetMatchingTools::overlapEnergyFraction (const std::vector <const reco::GenParticle*>& fObject, 
						const std::vector <const reco::GenParticle*>& fReference) const {
  if (fObject.empty()) return 0;
  double totalEnergy = 0;
  double overlapEnergy = 0;
  for (unsigned i = 0; i < fObject.size(); ++i) {
    totalEnergy += fObject [i]->energy();
    if (find (fReference.begin(), fReference.end(), fObject [i]) != fReference.end ()) overlapEnergy += fObject [i]->energy();
  }
  return overlapEnergy / totalEnergy;
}
