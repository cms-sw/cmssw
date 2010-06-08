#ifndef JetMatchingTools_h
#define JetMatchingTools_h

#include <vector>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace edm {
  class Event;
}
namespace reco {
  class CaloJet;
  class GenJet;
}
class CaloTower;
class CaloRecHit;
class DetId;
class PCaloHit; 

class JetMatchingTools {
 public:
  JetMatchingTools (const edm::Event& fEvent);
  ~JetMatchingTools ();

  /// get towers contributing to CaloJet
  std::vector <const CaloTower*> getConstituents (const reco::CaloJet& fJet ) ;
  /// get CaloRecHits contributing to the tower
  std::vector <const CaloRecHit*> getConstituents (const CaloTower& fTower) ;
  /// get cells contributing to the tower
  std::vector <DetId> getConstituentIds (const CaloTower& fTower) ;
  /// get PCaloHits contributing to the detId
  std::vector <const PCaloHit*> getPCaloHits (DetId fId) ;
  /// GEANT track ID
  int getTrackId (const PCaloHit& fHit) ;
  /// convert trackId to SimTrack
  const SimTrack* getTrack (unsigned fSimTrackId);
  /// Generator ID
  int generatorId (unsigned fSimTrackId) ;
  /// GenParticle
  const reco::GenParticle* getGenParticle (int fGeneratorId);
  /// GenParticles for CaloJet
  std::vector <const reco::GenParticle*> getGenParticles (const reco::CaloJet& fJet, bool fVerbose = true);
  /// GenParticles for GenJet
  std::vector <const reco::GenParticle*> getGenParticles (const reco::GenJet& fJet);

  // reverse propagation
  /// CaloSimHits
  std::vector <const PCaloHit*> getPCaloHits (int fGeneratorId);
  /// CaloRecHits
  std::vector <const CaloRecHit*> getCaloRecHits (int fGeneratorId);
  /// CaloTowers
  std::vector <const CaloTower*> getCaloTowers (int fGeneratorId);

  /// energy in broken links
  double lostEnergyFraction (const reco::CaloJet& fJet );

  /// energy overlap
  double overlapEnergyFraction (const std::vector <const reco::GenParticle*>& fObject, 
				const std::vector <const reco::GenParticle*>& fReference) const;


  const EBRecHitCollection* getEBRecHitCollection ();
  const EERecHitCollection* getEERecHitCollection ();
  const HBHERecHitCollection* getHBHERecHitCollection ();
  const HORecHitCollection* getHORecHitCollection ();
  const HFRecHitCollection* getHFRecHitCollection ();
  const edm::PCaloHitContainer* getEBSimHitCollection ();
  const edm::PCaloHitContainer* getEESimHitCollection ();
  const edm::PCaloHitContainer* getHcalSimHitCollection ();
  const edm::SimTrackContainer* getSimTrackCollection ();
  const edm::SimVertexContainer* getSimVertexCollection ();
  const reco::CandidateCollection* getGenParticlesCollection ();


  
 private:
  const edm::Event* mEvent;
  const EBRecHitCollection* mEBRecHitCollection;
  const EERecHitCollection* mEERecHitCollection;
  const HBHERecHitCollection* mHBHERecHitCollection;
  const HORecHitCollection* mHORecHitCollection;
  const HFRecHitCollection* mHFRecHitCollection;
  const edm::PCaloHitContainer* mEBSimHitCollection;
  const edm::PCaloHitContainer* mEESimHitCollection;
  const edm::PCaloHitContainer* mHcalSimHitCollection;
  const edm::SimTrackContainer* mSimTrackCollection;
  const edm::SimVertexContainer* mSimVertexCollection;
  const reco::CandidateCollection* mGenParticleCollection;
};

#endif
