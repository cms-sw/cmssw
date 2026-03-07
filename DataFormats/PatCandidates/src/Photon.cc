//
//

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include <cmath>

using pat::Photon;

/// default constructor
Photon::Photon()
    : PATObject<reco::Photon>(reco::Photon()),
      embeddedSuperCluster_(false),
      embeddedSeedCluster_(false),
      embeddedRecHits_(false),
      passElectronVeto_(false),
      hasPixelSeed_(false),
      seedEnergy_(0.0),
      eMax_(0.0),
      e2nd_(0.0),
      e3x3_(0.0),
      eTop_(0.0),
      eBottom_(0.0),
      eLeft_(0.0),
      eRight_(0.0),
      see_(-999.),
      spp_(-999.),
      sep_(-999.),
      maxDR_(-999.),
      maxDRDPhi_(-999.),
      maxDRDEta_(-999.),
      maxDRRawEnergy_(-999.),
      subClusRawE1_(-999.),
      subClusRawE2_(-999.),
      subClusRawE3_(-999.),
      subClusDPhi1_(-999.),
      subClusDPhi2_(-999.),
      subClusDPhi3_(-999.),
      subClusDEta1_(-999.),
      subClusDEta2_(-999.),
      subClusDEta3_(-999.),
      cryEta_(-999.),
      cryPhi_(-999),
      iEta_(-999),
      iPhi_(-999) {}

/// constructor from reco::Photon
Photon::Photon(const reco::Photon& aPhoton)
    : PATObject<reco::Photon>(aPhoton),
      embeddedSuperCluster_(false),
      embeddedSeedCluster_(false),
      embeddedRecHits_(false),
      passElectronVeto_(false),
      hasPixelSeed_(false),
      seedEnergy_(0.0),
      eMax_(0.0),
      e2nd_(0.0),
      e3x3_(0.0),
      eTop_(0.0),
      eBottom_(0.0),
      eLeft_(0.0),
      eRight_(0.0),
      see_(-999.),
      spp_(-999.),
      sep_(-999.),
      maxDR_(-999.),
      maxDRDPhi_(-999.),
      maxDRDEta_(-999.),
      maxDRRawEnergy_(-999.),
      subClusRawE1_(-999.),
      subClusRawE2_(-999.),
      subClusRawE3_(-999.),
      subClusDPhi1_(-999.),
      subClusDPhi2_(-999.),
      subClusDPhi3_(-999.),
      subClusDEta1_(-999.),
      subClusDEta2_(-999.),
      subClusDEta3_(-999.),
      cryEta_(-999.),
      cryPhi_(-999),
      iEta_(-999),
      iPhi_(-999) {}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::RefToBase<reco::Photon>& aPhotonRef)
    : PATObject<reco::Photon>(aPhotonRef),
      embeddedSuperCluster_(false),
      embeddedSeedCluster_(false),
      embeddedRecHits_(false),
      passElectronVeto_(false),
      hasPixelSeed_(false),
      seedEnergy_(0.0),
      eMax_(0.0),
      e2nd_(0.0),
      e3x3_(0.0),
      eTop_(0.0),
      eBottom_(0.0),
      eLeft_(0.0),
      eRight_(0.0),
      see_(-999.),
      spp_(-999.),
      sep_(-999.),
      maxDR_(-999.),
      maxDRDPhi_(-999.),
      maxDRDEta_(-999.),
      maxDRRawEnergy_(-999.),
      subClusRawE1_(-999.),
      subClusRawE2_(-999.),
      subClusRawE3_(-999.),
      subClusDPhi1_(-999.),
      subClusDPhi2_(-999.),
      subClusDPhi3_(-999.),
      subClusDEta1_(-999.),
      subClusDEta2_(-999.),
      subClusDEta3_(-999.),
      cryEta_(-999.),
      cryPhi_(-999),
      iEta_(-999),
      iPhi_(-999) {}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::Ptr<reco::Photon>& aPhotonRef)
    : PATObject<reco::Photon>(aPhotonRef),
      embeddedSuperCluster_(false),
      embeddedSeedCluster_(false),
      embeddedRecHits_(false),
      passElectronVeto_(false),
      hasPixelSeed_(false),
      seedEnergy_(0.0),
      eMax_(0.0),
      e2nd_(0.0),
      e3x3_(0.0),
      eTop_(0.0),
      eBottom_(0.0),
      eLeft_(0.0),
      eRight_(0.0),
      see_(-999.),
      spp_(-999.),
      sep_(-999.),
      maxDR_(-999.),
      maxDRDPhi_(-999.),
      maxDRDEta_(-999.),
      maxDRRawEnergy_(-999.),
      subClusRawE1_(-999.),
      subClusRawE2_(-999.),
      subClusRawE3_(-999.),
      subClusDPhi1_(-999.),
      subClusDPhi2_(-999.),
      subClusDPhi3_(-999.),
      subClusDEta1_(-999.),
      subClusDEta2_(-999.),
      subClusDEta3_(-999.),
      cryEta_(-999.),
      cryPhi_(-999),
      iEta_(-999),
      iPhi_(-999) {}

// Helper to create reco::Photon from scouting photon
namespace {
  reco::Photon makeRecoPhoton(const Run3ScoutingPhoton& sPhoton) {
    float px = sPhoton.pt() * std::cos(sPhoton.phi());
    float py = sPhoton.pt() * std::sin(sPhoton.phi());
    float pz = sPhoton.pt() * std::sinh(sPhoton.eta());
    float energy = std::sqrt(px * px + py * py + pz * pz + sPhoton.m() * sPhoton.m());
    reco::Photon::LorentzVector p4(px, py, pz, energy);
    reco::Photon::Point caloPos(0, 0, 0);
    reco::Photon::Point vtx(0, 0, 0);
    return reco::Photon(p4, caloPos, reco::PhotonCoreRef(), vtx);
  }
}  // namespace

/// constructor from Run3ScoutingPhoton
Photon::Photon(const Run3ScoutingPhoton& sPhoton)
    : PATObject<reco::Photon>(makeRecoPhoton(sPhoton)),
      embeddedSuperCluster_(false),
      embeddedSeedCluster_(false),
      embeddedRecHits_(false),
      passElectronVeto_(false),
      hasPixelSeed_(false),
      seedEnergy_(0.0),
      eMax_(0.0),
      e2nd_(0.0),
      e3x3_(0.0),
      eTop_(0.0),
      eBottom_(0.0),
      eLeft_(0.0),
      eRight_(0.0),
      see_(-999.),
      spp_(-999.),
      sep_(-999.),
      maxDR_(-999.),
      maxDRDPhi_(-999.),
      maxDRDEta_(-999.),
      maxDRRawEnergy_(-999.),
      subClusRawE1_(-999.),
      subClusRawE2_(-999.),
      subClusRawE3_(-999.),
      subClusDPhi1_(-999.),
      subClusDPhi2_(-999.),
      subClusDPhi3_(-999.),
      subClusDEta1_(-999.),
      subClusDEta2_(-999.),
      subClusDEta3_(-999.),
      cryEta_(-999.),
      cryPhi_(-999),
      iEta_(-999),
      iPhi_(-999) {
  isScoutingPhoton_ = true;

  // Store shower shape variables as userFloats
  this->addUserFloat("sigmaIetaIeta", sPhoton.sigmaIetaIeta());
  this->addUserFloat("hOverE", sPhoton.hOverE());
  this->addUserFloat("r9", sPhoton.r9());
  this->addUserFloat("sMin", sPhoton.sMin());
  this->addUserFloat("sMaj", sPhoton.sMaj());

  // Store energy variables
  this->addUserFloat("rawEnergy", sPhoton.rawEnergy());
  this->addUserFloat("preshowerEnergy", sPhoton.preshowerEnergy());
  this->addUserFloat("corrEcalEnergyError", sPhoton.corrEcalEnergyError());

  // Store isolation as userFloats (for NanoAOD access)
  this->addUserFloat("ecalIso", sPhoton.ecalIso());
  this->addUserFloat("hcalIso", sPhoton.hcalIso());
  this->addUserFloat("trkIso", sPhoton.trkIso());

  // Also set isolation using PAT isolation keys
  this->setIsolation(pat::TrackIso, sPhoton.trkIso());
  this->setIsolation(pat::EcalIso, sPhoton.ecalIso());
  this->setIsolation(pat::HcalIso, sPhoton.hcalIso());

  // Store scouting-specific ECAL crystal and rechit information
  scoutingSeedId_ = sPhoton.seedId();
  scoutingNClusters_ = sPhoton.nClusters();
  scoutingNCrystals_ = sPhoton.nCrystals();
  scoutingEnergyMatrix_ = sPhoton.energyMatrix();
  scoutingTimingMatrix_ = sPhoton.timingMatrix();
  scoutingDetIds_ = sPhoton.detIds();
  scoutingRechitZeroSuppression_ = sPhoton.rechitZeroSuppression();
}

/// destructor
Photon::~Photon() {}

std::ostream& reco::operator<<(std::ostream& out, const pat::Photon& obj) {
  if (!out)
    return out;

  out << "\tpat::Photon: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " << obj.energy() << "/" << obj.pt() << "/" << obj.eta() << "/" << obj.phi();
  return out;
}

/// override the superCluster method from CaloJet, to access the internal storage of the supercluster
/// this returns a transient Ref which *should never be persisted*!
reco::SuperClusterRef Photon::superCluster() const {
  if (embeddedSuperCluster_) {
    if (embeddedSeedCluster_ || !basicClusters_.empty() || !preshowerClusters_.empty()) {
      if (!superClusterRelinked_.isSet()) {
        std::unique_ptr<std::vector<reco::SuperCluster> > sc(new std::vector<reco::SuperCluster>(superCluster_));
        if (embeddedSeedCluster_ && !(*sc)[0].seed().isAvailable()) {
          (*sc)[0].setSeed(seed());
        }
        if (!basicClusters_.empty() && !(*sc)[0].clusters().isAvailable()) {
          reco::CaloClusterPtrVector clusters;
          for (unsigned int iclus = 0; iclus < basicClusters_.size(); ++iclus) {
            clusters.push_back(reco::CaloClusterPtr(&basicClusters_, iclus));
          }
          (*sc)[0].setClusters(clusters);
        }
        if (!preshowerClusters_.empty() && !(*sc)[0].preshowerClusters().isAvailable()) {
          reco::CaloClusterPtrVector clusters;
          for (unsigned int iclus = 0; iclus < preshowerClusters_.size(); ++iclus) {
            clusters.push_back(reco::CaloClusterPtr(&preshowerClusters_, iclus));
          }
          (*sc)[0].setPreshowerClusters(clusters);
        }
        superClusterRelinked_.set(std::move(sc));
      }
      return reco::SuperClusterRef(&*superClusterRelinked_, 0);
    } else {
      return reco::SuperClusterRef(&superCluster_, 0);
    }
  } else {
    return reco::Photon::superCluster();
  }
}

/// direct access to the seed cluster
reco::CaloClusterPtr Photon::seed() const {
  if (embeddedSeedCluster_) {
    return reco::CaloClusterPtr(&seedCluster_, 0);
  } else {
    return reco::Photon::superCluster()->seed();
  }
}

/// method to store the photon's supercluster internally
void Photon::embedSuperCluster() {
  superCluster_.clear();
  if (reco::Photon::superCluster().isNonnull()) {
    superCluster_.push_back(*reco::Photon::superCluster());
    embeddedSuperCluster_ = true;
  }
}

/// Stores the electron's SeedCluster (reco::BasicClusterPtr) internally
void Photon::embedSeedCluster() {
  seedCluster_.clear();
  if (reco::Photon::superCluster().isNonnull() && reco::Photon::superCluster()->seed().isNonnull()) {
    seedCluster_.push_back(*reco::Photon::superCluster()->seed());
    embeddedSeedCluster_ = true;
  }
}

/// Stores the electron's BasicCluster (reco::CaloCluster) internally
void Photon::embedBasicClusters() {
  basicClusters_.clear();
  if (reco::Photon::superCluster().isNonnull()) {
    reco::CaloCluster_iterator itscl = reco::Photon::superCluster()->clustersBegin();
    reco::CaloCluster_iterator itsclE = reco::Photon::superCluster()->clustersEnd();
    for (; itscl != itsclE; ++itscl) {
      basicClusters_.push_back(**itscl);
    }
  }
}

/// Stores the electron's PreshowerCluster (reco::CaloCluster) internally
void Photon::embedPreshowerClusters() {
  preshowerClusters_.clear();
  if (reco::Photon::superCluster().isNonnull()) {
    reco::CaloCluster_iterator itscl = reco::Photon::superCluster()->preshowerClustersBegin();
    reco::CaloCluster_iterator itsclE = reco::Photon::superCluster()->preshowerClustersEnd();
    for (; itscl != itsclE; ++itscl) {
      preshowerClusters_.push_back(**itscl);
    }
  }
}

// method to store the RecHits internally
void Photon::embedRecHits(const EcalRecHitCollection* rechits) {
  if (rechits != nullptr) {
    recHits_ = *rechits;
    embeddedRecHits_ = true;
  }
}

// method to retrieve a photon ID (or throw)
Bool_t Photon::photonID(const std::string& name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name)
      return it->second;
  }
  cms::Exception ex("Key not found");
  ex << "pat::Photon: the ID " << name << " can't be found in this pat::Photon.\n";
  ex << "The available IDs are: ";
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    ex << "'" << it->first << "' ";
  }
  ex << ".\n";
  throw ex;
}
// check if an ID is there
bool Photon::isPhotonIDAvailable(const std::string& name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name)
      return true;
  }
  return false;
}

edm::RefVector<pat::PackedCandidateCollection> Photon::associatedPackedPFCandidates() const {
  edm::RefVector<pat::PackedCandidateCollection> ret(packedPFCandidates_.id());
  for (uint16_t idx : associatedPackedFCandidateIndices_) {
    ret.push_back(edm::Ref<pat::PackedCandidateCollection>(packedPFCandidates_, idx));
  }
  return ret;
}

/// Returns the reference to the parent PF candidate with index i.
/// For use in TopProjector.
reco::CandidatePtr Photon::sourceCandidatePtr(size_type i) const {
  if (i >= associatedPackedFCandidateIndices_.size()) {
    return reco::CandidatePtr();
  } else {
    return reco::CandidatePtr(edm::refToPtr(
        edm::Ref<pat::PackedCandidateCollection>(packedPFCandidates_, associatedPackedFCandidateIndices_[i])));
  }
}

// ---- Universal accessors with scouting dispatch ----

uint32_t Photon::seedId() const {
  if (isScoutingPhoton_) {
    return scoutingSeedId_;
  }
  auto sc = superCluster();
  if (sc.isNonnull() && sc->seed().isNonnull()) {
    return sc->seed()->seed().rawId();
  }
  return 0;
}

uint32_t Photon::nClusters() const {
  if (isScoutingPhoton_) {
    return scoutingNClusters_;
  }
  auto sc = superCluster();
  return sc.isNonnull() ? sc->clustersSize() : 0;
}

uint32_t Photon::nCrystals() const {
  if (isScoutingPhoton_) {
    return scoutingNCrystals_;
  }
  auto sc = superCluster();
  return sc.isNonnull() ? sc->size() : 0;
}
