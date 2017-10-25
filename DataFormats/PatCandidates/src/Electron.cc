//
//

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include <limits>

using namespace pat;

/// default constructor
Electron::Electron() :
    Lepton<reco::GsfElectron>(),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedPflowSuperCluster_(false),
    embeddedTrack_(false),
    embeddedSeedCluster_(false),
    embeddedRecHits_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(Candidate::LorentzVector(0.,0.,0.,0.)),
    ecalRegressionEnergy_(0.0),
    ecalTrackRegressionEnergy_(0.0),
    ecalRegressionError_(0.0),
    ecalTrackRegressionError_(0.0),
    ecalScale_(-99999.),
    ecalSmear_(-99999.),
    ecalRegressionScale_(-99999.),
    ecalRegressionSmear_(-99999.),
    ecalTrackRegressionScale_(-99999.),
    ecalTrackRegressionSmear_(-99999.),
    packedPFCandidates_(),
    associatedPackedFCandidateIndices_()
{
  initImpactParameters();
}

/// constructor from reco::GsfElectron
Electron::Electron(const reco::GsfElectron & anElectron) :
    Lepton<reco::GsfElectron>(anElectron),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedPflowSuperCluster_(false),
    embeddedTrack_(false),
    embeddedSeedCluster_(false),
    embeddedRecHits_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectron.p4())
{
  initImpactParameters();
}

/// constructor from a RefToBase to a reco::GsfElectron (to be superseded by Ptr counterpart)
Electron::Electron(const edm::RefToBase<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedPflowSuperCluster_(false),
    embeddedTrack_(false), 
    embeddedSeedCluster_(false),
    embeddedRecHits_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectronRef->p4())
{
  initImpactParameters();
}

/// constructor from a Ptr to a reco::GsfElectron
Electron::Electron(const edm::Ptr<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedPflowSuperCluster_(false),
    embeddedTrack_(false),
    embeddedSeedCluster_(false),
    embeddedRecHits_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectronRef->p4())
{
  initImpactParameters();
}

/// destructor
Electron::~Electron() {
}

/// pipe operator (introduced to use pat::Electron with PFTopProjectors)
std::ostream& 
reco::operator<<(std::ostream& out, const pat::Electron& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Electron: ";
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

/// initializes the impact parameter container vars
void Electron::initImpactParameters() {
  std::fill(ip_, ip_+IpTypeSize, 0.0f);
  std::fill(eip_, eip_+IpTypeSize, 0.0f);
  cachedIP_ = 0;
}


/// override the reco::GsfElectron::gsfTrack method, to access the internal storage of the supercluster
reco::GsfTrackRef Electron::gsfTrack() const {
  if (embeddedGsfTrack_) {
    return reco::GsfTrackRef(&gsfTrack_, 0);
  } else {
    return reco::GsfElectron::gsfTrack();
  }
}

/// override the virtual reco::GsfElectron::core method, so that the embedded core can be used by GsfElectron client methods
reco::GsfElectronCoreRef Electron::core() const {
  if (embeddedGsfElectronCore_) {
    return reco::GsfElectronCoreRef(&gsfElectronCore_, 0);
  } else {
    return reco::GsfElectron::core();
  }
}


/// override the reco::GsfElectron::superCluster method, to access the internal storage of the supercluster
reco::SuperClusterRef Electron::superCluster() const {
  if (embeddedSuperCluster_) {
    if (embeddedSeedCluster_ || !basicClusters_.empty() || !preshowerClusters_.empty()) {
        if (!superClusterRelinked_.isSet()) {
            std::unique_ptr<std::vector<reco::SuperCluster> > sc(new std::vector<reco::SuperCluster>(superCluster_));
            if (embeddedSeedCluster_ && !(*sc)[0].seed().isAvailable()) {
                (*sc)[0].setSeed(seed());
            }
            if (!basicClusters_.empty() && !(*sc)[0].clusters().isAvailable()) {
                reco::CaloClusterPtrVector clusters;
                for (unsigned int iclus=0; iclus<basicClusters_.size(); ++iclus) {
                    clusters.push_back(reco::CaloClusterPtr(&basicClusters_,iclus));
                }
                (*sc)[0].setClusters(clusters);
            }
            if (!preshowerClusters_.empty() && !(*sc)[0].preshowerClusters().isAvailable()) {
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
    //relink caloclusters if needed
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return reco::GsfElectron::superCluster();
  }
}

/// override the reco::GsfElectron::pflowSuperCluster method, to access the internal storage of the supercluster
reco::SuperClusterRef Electron::parentSuperCluster() const {
  if (embeddedPflowSuperCluster_) {
    return reco::SuperClusterRef(&pflowSuperCluster_, 0);
  } else {
    return reco::GsfElectron::parentSuperCluster();
  }
}

/// direct access to the seed cluster
reco::CaloClusterPtr Electron::seed() const {
  if(embeddedSeedCluster_){
    return reco::CaloClusterPtr(&seedCluster_,0);
  } else {
    return reco::GsfElectron::superCluster()->seed();
  }
}

/// override the reco::GsfElectron::closestCtfTrack method, to access the internal storage of the track
reco::TrackRef Electron::closestCtfTrackRef() const {
  if (embeddedTrack_) {
    return reco::TrackRef(&track_, 0);
  } else {
    return reco::GsfElectron::closestCtfTrackRef();
  }
}

// the name of the method is misleading, users should use gsfTrack of closestCtfTrack
reco::TrackRef Electron::track() const {
  return reco::TrackRef();
}

/// Stores the electron's core (reco::GsfElectronCoreRef) internally
void Electron::embedGsfElectronCore() {
  gsfElectronCore_.clear();
  if (reco::GsfElectron::core().isNonnull()) {
      gsfElectronCore_.push_back(*reco::GsfElectron::core());
      embeddedGsfElectronCore_ = true;
  }
}

/// Stores the electron's gsfTrack (reco::GsfTrackRef) internally
void Electron::embedGsfTrack() {
  gsfTrack_.clear();
  if (reco::GsfElectron::gsfTrack().isNonnull()) {
      gsfTrack_.push_back(*reco::GsfElectron::gsfTrack());
      embeddedGsfTrack_ = true;
  }
}


/// Stores the electron's SuperCluster (reco::SuperClusterRef) internally
void Electron::embedSuperCluster() {
  superCluster_.clear();
  if (reco::GsfElectron::superCluster().isNonnull()) {
      superCluster_.push_back(*reco::GsfElectron::superCluster());
      embeddedSuperCluster_ = true;
  }
}

/// Stores the electron's SuperCluster (reco::SuperClusterRef) internally
void Electron::embedPflowSuperCluster() {
  pflowSuperCluster_.clear();
  if (reco::GsfElectron::parentSuperCluster().isNonnull()) {
      pflowSuperCluster_.push_back(*reco::GsfElectron::parentSuperCluster());
      embeddedPflowSuperCluster_ = true;
  }
}

/// Stores the electron's SeedCluster (reco::BasicClusterPtr) internally
void Electron::embedSeedCluster() {
  seedCluster_.clear();
  if (reco::GsfElectron::superCluster().isNonnull() && reco::GsfElectron::superCluster()->seed().isNonnull()) {
    seedCluster_.push_back(*reco::GsfElectron::superCluster()->seed());
    embeddedSeedCluster_ = true;
  }
}

/// Stores the electron's BasicCluster (reco::CaloCluster) internally
void Electron::embedBasicClusters() {
  basicClusters_.clear();
  if (reco::GsfElectron::superCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::GsfElectron::superCluster()->clustersBegin();
    reco::CaloCluster_iterator itsclE = reco::GsfElectron::superCluster()->clustersEnd();
    for(;itscl!=itsclE;++itscl){
      basicClusters_.push_back( **itscl ) ;
    } 
  }
}

/// Stores the electron's PreshowerCluster (reco::CaloCluster) internally
void Electron::embedPreshowerClusters() {
  preshowerClusters_.clear();
  if (reco::GsfElectron::superCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::GsfElectron::superCluster()->preshowerClustersBegin();
    reco::CaloCluster_iterator itsclE = reco::GsfElectron::superCluster()->preshowerClustersEnd();
    for(;itscl!=itsclE;++itscl){
      preshowerClusters_.push_back( **itscl ) ;
    }
  }
}

/// Stores the electron's PflowBasicCluster (reco::CaloCluster) internally
void Electron::embedPflowBasicClusters() {
  pflowBasicClusters_.clear();
  if (reco::GsfElectron::parentSuperCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::GsfElectron::parentSuperCluster()->clustersBegin();
    reco::CaloCluster_iterator itsclE = reco::GsfElectron::parentSuperCluster()->clustersEnd();
    for(;itscl!=itsclE;++itscl){
      pflowBasicClusters_.push_back( **itscl ) ;
    }
  }
}

/// Stores the electron's PflowPreshowerCluster (reco::CaloCluster) internally
void Electron::embedPflowPreshowerClusters() {
  pflowPreshowerClusters_.clear();
  if (reco::GsfElectron::parentSuperCluster().isNonnull()){
    reco::CaloCluster_iterator itscl = reco::GsfElectron::parentSuperCluster()->preshowerClustersBegin();
    reco::CaloCluster_iterator itsclE = reco::GsfElectron::parentSuperCluster()->preshowerClustersEnd();
    for(;itscl!=itsclE;++itscl){
      pflowPreshowerClusters_.push_back( **itscl ) ;
    }
  }
}

/// method to store the electron's track internally
void Electron::embedTrack() {
  track_.clear();
  if (reco::GsfElectron::closestCtfTrackRef().isNonnull()) {
      track_.push_back(*reco::GsfElectron::closestCtfTrackRef());
      embeddedTrack_ = true;
  }
}

// method to store the RecHits internally
void Electron::embedRecHits(const EcalRecHitCollection * rechits) {
  if (rechits!=nullptr) {
    recHits_ = *rechits;
    embeddedRecHits_ = true;
  }
}

/// Returns a specific electron ID associated to the pat::Electron given its name
/// For cut-based IDs, the value map has the following meaning:
/// 0: fails,
/// 1: passes electron ID only,
/// 2: passes electron Isolation only,
/// 3: passes electron ID and Isolation only,
/// 4: passes conversion rejection,
/// 5: passes conversion rejection and ID,
/// 6: passes conversion rejection and Isolation,
/// 7: passes the whole selection.
/// For more details have a look at:
/// https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
/// https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCategoryBasedElectronID
/// Note: an exception is thrown if the specified ID is not available
float Electron::electronID(const std::string& name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return it->second;
    }
    cms::Exception ex("Key not found");
    ex << "pat::Electron: the ID " << name << " can't be found in this pat::Electron.\n";
    ex << "The available IDs are: ";
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        ex << "'" << it->first << "' ";
    }
    ex << ".\n";
    throw ex;
}

/// Checks if a specific electron ID is associated to the pat::Electron.
bool Electron::isElectronIDAvailable(const std::string& name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return true;
    }
    return false;
}


/// reference to the source PFCandidates
reco::PFCandidateRef Electron::pfCandidateRef() const {
  if (embeddedPFCandidate_) {
    return reco::PFCandidateRef(&pfCandidate_, 0);
  } else {
    return pfCandidateRef_;
  }
}

/// Stores the PFCandidate pointed to by pfCandidateRef_ internally
void Electron::embedPFCandidate() {
  pfCandidate_.clear();
  if ( pfCandidateRef_.isAvailable() && pfCandidateRef_.isNonnull()) {
    pfCandidate_.push_back( *pfCandidateRef_ );
    embeddedPFCandidate_ = true;
  }
}

/// Returns the reference to the parent PF candidate with index i.
/// For use in TopProjector.
reco::CandidatePtr Electron::sourceCandidatePtr( size_type i ) const {
    if (pfCandidateRef_.isNonnull()) {
        if (i == 0) {
            return reco::CandidatePtr(edm::refToPtr(pfCandidateRef_));
        } else {
            i--;
        }
    }
    if (i >= associatedPackedFCandidateIndices_.size()) {
        return reco::CandidatePtr();
    } else {
        return reco::CandidatePtr(edm::refToPtr(edm::Ref<pat::PackedCandidateCollection>(packedPFCandidates_, associatedPackedFCandidateIndices_[i])));
    }
}



/// dB gives the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline.
///
/// IpType defines the type of the impact parameter
/// None is default and reverts to the old functionality.
///
/// Example: electron->dB(pat::Electron::PV2D)
/// will return the electron transverse impact parameter
/// relative to the primary vertex.
double Electron::dB(IpType type_) const {
  // more IP types (new)
  if ( cachedIP_ & (1 << int(type_))) {
    return ip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }
}

/// edB gives the uncertainty on the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline. 
///
/// IpType defines the type of the impact parameter
/// None is default and reverts to the old functionality.
///
/// Example: electron->edB(pat::Electron::PV2D)
/// will return the electron transverse impact parameter uncertainty
/// relative to the primary vertex.
double Electron::edB(IpType type_) const {
  // more IP types (new)
  if ( cachedIP_ & (1 << int(type_))) {
    return eip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }
}

/// Sets the impact parameter and its error wrt the beamline and caches it.
void Electron::setDB(double dB, double edB, IpType type){
  ip_[type] = dB; eip_[type] = edB; cachedIP_ |= (1 << int(type));
}

/// Set additional missing mva input variables for new mva ID (71X update)
void Electron::setMvaVariables( double sigmaIetaIphi, double ip3d){
  sigmaIetaIphi_ = sigmaIetaIphi;
  ip3d_ = ip3d;
} 

edm::RefVector<pat::PackedCandidateCollection> Electron::associatedPackedPFCandidates() const {
    edm::RefVector<pat::PackedCandidateCollection> ret(packedPFCandidates_.id());
    for (uint16_t idx : associatedPackedFCandidateIndices_) {
        ret.push_back(edm::Ref<pat::PackedCandidateCollection>(packedPFCandidates_, idx));
    }
    return ret;
}


