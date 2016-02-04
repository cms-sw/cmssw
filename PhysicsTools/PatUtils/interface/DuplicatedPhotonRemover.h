#ifndef PhysicsTools_PatUtils_DuplicatedPhotonRemover_h
#define PhysicsTools_PatUtils_DuplicatedPhotonRemover_h

#include "PhysicsTools/PatUtils/interface/GenericDuplicateRemover.h"
#include "PhysicsTools/PatUtils/interface/GenericOverlapFinder.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/View.h"

#include "CommonTools/Utils/interface/EtComparator.h"

namespace pat { 
    
    class DuplicatedPhotonRemover {

        public:
            // Checks if two objects share the same supercluster seed
            struct EqualBySuperClusterSeed {
                template<typename T1, typename T2>
                    bool operator()(const T1 &t1, const T2 &t2) const { 
                        return (t1.superCluster()->seed() == t2.superCluster()->seed());
                    }
            };

            // Checks if two objects share the same supercluster seed
            struct EqualBySuperCluster {
                template<typename T1, typename T2>
                    bool operator()(const T1 &t1, const T2 &t2) const { 
                        return (t1.superCluster() == t2.superCluster());
                    }
            };

            /// Indices of duplicated photons (same supercluster) to remove. It keeps the photons with highest energy.
            /// PhotonCollection can be anything that has a "begin()" and "end()", and that hold things which have a "superCluster()" method
            /// notable examples are std::vector<Photon> and edm::View<Photon> (but GsfElectrons work too)
            template <typename PhotonCollection> 
            std::auto_ptr< std::vector<size_t> > duplicatesBySuperCluster(const PhotonCollection &photons) const ;
            
            /// Indices of duplicated photons (same supercluster) to remove. It keeps the photons with highest energy.
            /// PhotonCollection can be anything that has a "begin()" and "end()", and that hold things which have a "superCluster()" method
            /// notable examples are std::vector<Photon> and edm::View<Photon> (but GsfElectrons work too)
            template <typename PhotonCollection> 
            std::auto_ptr< std::vector<size_t> > duplicatesBySeed(const PhotonCollection &photons) const ;

            /// Indices of photons which happen to be also electrons (that is, they share the same SC seed)
            template <typename PhotonCollection, typename ElectronCollection> 
            std::auto_ptr< pat::OverlapList > 
            electronsBySeed(const PhotonCollection &photons, const ElectronCollection &electrons) const ;

            /// Indices of photons which happen to be also electrons (that is, they share the same SC)
            template <typename PhotonCollection, typename ElectronCollection> 
            std::auto_ptr< pat::OverlapList > 
            electronsBySuperCluster(const PhotonCollection &photons, const ElectronCollection &electrons) const ;

            // ===== Concrete versions for users (and to get it compiled, so I can see if there are errors) ===
            std::auto_ptr< std::vector<size_t> > duplicatesBySeed(const reco::PhotonCollection &photons) const ;
            std::auto_ptr< std::vector<size_t> > duplicatesBySeed(const edm::View<reco::Photon> &photons) const ;
            std::auto_ptr< std::vector<size_t> > duplicatesBySuperCluster(const edm::View<reco::Photon> &photons) const ;
            std::auto_ptr< std::vector<size_t> > duplicatesBySuperCluster(const reco::PhotonCollection &photons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySeed(const reco::PhotonCollection &photons, 
                    const reco::GsfElectronCollection electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySeed(const edm::View<reco::Photon> &photons, 
                    const reco::GsfElectronCollection electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySuperCluster(const edm::View<reco::Photon> &photons, 
                    const reco::GsfElectronCollection electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySuperCluster(const reco::PhotonCollection &photons, 
                    const reco::GsfElectronCollection electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySeed(const reco::PhotonCollection &photons, 
                    const edm::View<reco::GsfElectron> electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySeed(const edm::View<reco::Photon> &photons, 
                    const edm::View<reco::GsfElectron> electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySuperCluster(const edm::View<reco::Photon> &photons, 
                    const edm::View<reco::GsfElectron> electrons) const ;
            std::auto_ptr< pat::OverlapList > electronsBySuperCluster(const reco::PhotonCollection &photons, 
                    const edm::View<reco::GsfElectron> electrons) const ;
    };
}

template<typename PhotonCollection>
std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySuperCluster(const PhotonCollection &photons) const {
    typedef typename PhotonCollection::value_type PhotonType;
    pat::GenericDuplicateRemover<EqualBySuperCluster, GreaterByEt<PhotonType> > dups;
    return dups.duplicates(photons);
}

template<typename PhotonCollection>
std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedPhotonRemover::duplicatesBySeed(const PhotonCollection &photons) const {
    typedef typename PhotonCollection::value_type PhotonType;
    pat::GenericDuplicateRemover<EqualBySuperClusterSeed, GreaterByEt<PhotonType> > dups;
    return dups.duplicates(photons);
}

/// Indices of photons which happen to be also electrons (that is, they share the same SC)
template <typename PhotonCollection, typename ElectronCollection> 
std::auto_ptr< pat::OverlapList > 
pat::DuplicatedPhotonRemover::electronsBySuperCluster(const PhotonCollection &photons, const ElectronCollection &electrons) const {
    pat::GenericOverlapFinder< pat::OverlapDistance<EqualBySuperCluster> > ovl;
    return ovl.find(photons, electrons);
}

/// Indices of photons which happen to be also electrons (that is, they share the same SC)
template <typename PhotonCollection, typename ElectronCollection> 
std::auto_ptr< pat::OverlapList > 
pat::DuplicatedPhotonRemover::electronsBySeed(const PhotonCollection &photons, const ElectronCollection &electrons) const {
    pat::GenericOverlapFinder< pat::OverlapDistance<EqualBySuperClusterSeed> > ovl;
    return ovl.find(photons, electrons);
}

#endif
