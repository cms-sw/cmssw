#ifndef JetMatchingTools_h
#define JetMatchingTools_h

#include <vector>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

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
  struct JetConstituent {
    DetId id;
    double energy;

    JetConstituent() {}
    ~JetConstituent() {}
    JetConstituent(const JetConstituent &j) : id(j.id), energy(j.energy) {}
    JetConstituent(const EcalRecHit &ehit) : id(ehit.detid()), energy(ehit.energy()) {}
    JetConstituent(const CaloRecHit &ehit) : id(ehit.detid()), energy(ehit.energy()) {}
  };

  JetMatchingTools (const edm::Event& fEvent, edm::ConsumesCollector&& iC );
  ~JetMatchingTools ();

  /// get towers contributing to CaloJet
  std::vector <const CaloTower*> getConstituents (const reco::CaloJet& fJet ) ;
  /// get CaloRecHits contributing to the tower
  std::vector <JetConstituent> getConstituentHits(const CaloTower& fTower);
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

  edm::EDGetTokenT<EBRecHitCollection> input_ebrechits_token_;
  edm::EDGetTokenT<EERecHitCollection> input_eerechits_token_;
  edm::EDGetTokenT<HBHERecHitCollection> input_hbherechits_token_;
  edm::EDGetTokenT<HORecHitCollection> input_horechits_token_;
  edm::EDGetTokenT<HFRecHitCollection> input_hfrechits_token_;
  edm::EDGetTokenT<edm::PCaloHitContainer> input_pcalohits_eecal_token_;
  edm::EDGetTokenT<edm::PCaloHitContainer> input_pcalohits_ebcal_token_;
  edm::EDGetTokenT<edm::PCaloHitContainer> input_pcalohits_hcal_token_;
  edm::EDGetTokenT<edm::SimTrackContainer> input_simtrack_token_;
  edm::EDGetTokenT<edm::SimVertexContainer> input_simvertex_token_;
  edm::EDGetTokenT<reco::CandidateCollection> input_cands_token_;

};

#endif
