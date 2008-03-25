#include "PhysicsTools/PatUtils/interface/DuplicatedPhotonRemover.h"

#include <algorithm>

std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySeed(const reco::PhotonCollection &photons) const {
    return duplicatesBySeed<reco::PhotonCollection>(photons);
}

std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySeed(const edm::View<reco::Photon> &photons) const {
    return duplicatesBySeed< edm::View<reco::Photon> >(photons);
}

std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySuperCluster(const reco::PhotonCollection &photons) const {
    return duplicatesBySuperCluster<reco::PhotonCollection>(photons);
}

std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySuperCluster(const edm::View<reco::Photon> &photons) const {
    return duplicatesBySuperCluster< edm::View<reco::Photon> >(photons);
}

// ================ ELECTRONS  =============================
// ---------------- against EleCollection  -----------------------------
std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySeed(const reco::PhotonCollection &photons, 
        const reco::GsfElectronCollection electrons) const {
    return electronsBySeed<reco::PhotonCollection, reco::GsfElectronCollection>(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySeed(const edm::View<reco::Photon> &photons, 
        const reco::GsfElectronCollection electrons) const {
    return electronsBySeed<edm::View<reco::Photon>, reco::GsfElectronCollection>(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySuperCluster(const edm::View<reco::Photon> &photons, 
        const reco::GsfElectronCollection electrons) const {
    return electronsBySuperCluster<edm::View<reco::Photon>, reco::GsfElectronCollection>(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySuperCluster(const reco::PhotonCollection &photons, 
        const reco::GsfElectronCollection  electrons) const {
    return electronsBySuperCluster<reco::PhotonCollection, reco::GsfElectronCollection>(photons, electrons);
}

// ---------------- against EleView  -----------------------------
std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySeed(const reco::PhotonCollection &photons, 
        const edm::View<reco::GsfElectron>  electrons) const {
    return electronsBySeed<reco::PhotonCollection, edm::View<reco::GsfElectron> >(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySeed(const edm::View<reco::Photon> &photons, 
        const edm::View<reco::GsfElectron>  electrons) const {
    return electronsBySeed<edm::View<reco::Photon>, edm::View<reco::GsfElectron> >(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySuperCluster(const edm::View<reco::Photon> &photons, 
        const edm::View<reco::GsfElectron>  electrons) const {
    return electronsBySuperCluster<edm::View<reco::Photon>, edm::View<reco::GsfElectron> >(photons, electrons);
}

std::auto_ptr< pat::OverlapList >
pat::DuplicatedPhotonRemover::electronsBySuperCluster(const reco::PhotonCollection &photons, 
        const edm::View<reco::GsfElectron>  electrons) const {
    return electronsBySuperCluster<reco::PhotonCollection, edm::View<reco::GsfElectron> >(photons, electrons);
}

