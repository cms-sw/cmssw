//
//

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/Common/interface/RefToPtr.h"


using pat::Photon;


/// default constructor
Photon::Photon() :
    PATObject<reco::Photon>(reco::Photon()),
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
    iPhi_(-999)
{
}

/// constructor from reco::Photon
Photon::Photon(const reco::Photon & aPhoton) :
    PATObject<reco::Photon>(aPhoton),
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
    iPhi_(-999)
{
}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::RefToBase<reco::Photon> & aPhotonRef) :
    PATObject<reco::Photon>(aPhotonRef),
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
    iPhi_(-999)
{
}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::Ptr<reco::Photon> & aPhotonRef) :
    PATObject<reco::Photon>(aPhotonRef),
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
    iPhi_(-999)
{
}

/// destructor
Photon::~Photon() {
}

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Photon& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Photon: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
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
            if (basicClusters_.size() && !(*sc)[0].clusters().isAvailable()) {
                reco::CaloClusterPtrVector clusters;
                for (unsigned int iclus=0; iclus<basicClusters_.size(); ++iclus) {
                    clusters.push_back(reco::CaloClusterPtr(&basicClusters_,iclus));
                }
                (*sc)[0].setClusters(clusters);
            }
            if (preshowerClusters_.size() && !(*sc)[0].preshowerClusters().isAvailable()) {
                reco::CaloClusterPtrVector clusters;
                for (unsigned int iclus=0; iclus<preshowerClusters_.size(); ++iclus) {
                    clusters.push_back(reco::CaloClusterPtr(&preshowerClusters_,iclus));
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
  if(embeddedSeedCluster_){
    return reco::CaloClusterPtr(&seedCluster_,0);
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
  if (reco::Photon::superCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::Photon::superCluster()->clustersBegin();
    reco::CaloCluster_iterator itsclE = reco::Photon::superCluster()->clustersEnd();
    for(;itscl!=itsclE;++itscl){
      basicClusters_.push_back( **itscl ) ;
    } 
  }
}

/// Stores the electron's PreshowerCluster (reco::CaloCluster) internally
void Photon::embedPreshowerClusters() {
  preshowerClusters_.clear();
  if (reco::Photon::superCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::Photon::superCluster()->preshowerClustersBegin();
    reco::CaloCluster_iterator itsclE = reco::Photon::superCluster()->preshowerClustersEnd();
    for(;itscl!=itsclE;++itscl){
      preshowerClusters_.push_back( **itscl ) ;
    }
  }
}

// method to store the RecHits internally
void Photon::embedRecHits(const EcalRecHitCollection * rechits) {
  if (rechits!=0) {
    recHits_ = *rechits;
    embeddedRecHits_ = true;
  }
}

// method to retrieve a photon ID (or throw)
Bool_t Photon::photonID(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name) return it->second;
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
bool Photon::isPhotonIDAvailable(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name) return true;
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
reco::CandidatePtr Photon::sourceCandidatePtr( size_type i ) const {
    if (i >= associatedPackedFCandidateIndices_.size()) {
        return reco::CandidatePtr();
    } else {
        return reco::CandidatePtr(edm::refToPtr(edm::Ref<pat::PackedCandidateCollection>(packedPFCandidates_, associatedPackedFCandidateIndices_[i])));
    }
}

