#ifndef L1Trigger_Phase2L1ParticleFlow_CALOCLUSTERER_H
#define L1Trigger_Phase2L1ParticleFlow_CALOCLUSTERER_H
/** 
 * Classes for calorimetric re-clustering
 * */

// fwd declarations
namespace edm { class ParameterSet; }

// real includes
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCluster.h"

namespace l1tpf_calo {
    class Grid {
        public:
            virtual ~Grid() {}
            unsigned int size() const { return ncells_; }
            virtual int find_cell(float eta, float phi) const = 0;
            int neighbour(int icell, unsigned int idx) const { return neighbours_[icell][idx]; }
            float eta(int icell) const { return eta_[icell]; }
            float phi(int icell) const { return phi_[icell]; }
            float etaWidth(int icell) const { return etaWidth_[icell]; }
            float phiWidth(int icell) const { return phiWidth_[icell]; }
            int ieta(int icell) const { return ieta_[icell]; }
            int iphi(int icell) const { return iphi_[icell]; }
        protected:
            Grid(unsigned int size) : ncells_(size), eta_(size), etaWidth_(size), phi_(size), phiWidth_(size), ieta_(size), iphi_(size), neighbours_(size) {}
            unsigned int ncells_;
            std::vector<float> eta_, etaWidth_, phi_, phiWidth_;
            std::vector<int> ieta_, iphi_;
            std::vector<std::array<int,8>> neighbours_; // indices of the neigbours, -1 = none
    };

    class Phase1Grid : public Grid {
        public:
            Phase1Grid() ;
            virtual int find_cell(float eta, float phi) const override ;
            int ifind_cell(int ieta, int iphi) const { return cell_map_[(ieta+nEta_) + 2*nEta_*(iphi-1)]; }
        protected:
            static const int nEta_ = 41, nPhi_ = 72, ietaCoarse_ = 29, ietaVeryCoarse_ = 40;
            static const float towerEtas_[nEta_];
            std::vector<int> cell_map_;
            // valid ieta, iphi (does not check for outside bounds, only for non-existence of ieta=0, iphi=0, and coarser towers at high eta)
            bool valid_ieta_iphi(int ieta, int iphi) const {
                if (ieta == 0 || iphi == 0) return false;
                if (std::abs(ieta) >= ietaVeryCoarse_ && (iphi % 4 != 1)) return false;
                if (std::abs(ieta) >= ietaCoarse_     && (iphi % 2 != 1)) return false;
                return true;
            }
            // move by +/-1 around a cell; return icell or -1 if not available
            int imove(int ieta, int iphi, int deta, int dphi) ;
            
    };

    template<typename T>
    class GridData {
        public:
            GridData() : grid_(nullptr), data_(), empty_() {}
            GridData(const Grid &grid) : grid_(&grid), data_(grid.size()), empty_() {}
        
            T & operator()(float eta, float phi) { return data_[grid_->find_cell(eta,phi)]; }
            const T & operator()(float eta, float phi) const { return data_[grid_->find_cell(eta,phi)]; }
            
            const Grid & grid() const { return *grid_; }

            unsigned int size() const { return data_.size(); }            

            float eta(int icell) const { return grid().eta(icell); }
            float phi(int icell) const { return grid().phi(icell); }
            int ieta(int icell) const { return grid().ieta(icell); }
            int iphi(int icell) const { return grid().iphi(icell); }

            T & operator[](int icell) { return data_[icell]; }
            const T & operator[](int icell) const { return data_[icell]; }

            const T & neigh(int icell, unsigned int idx) const { 
                int ineigh = grid_->neighbour(icell, idx);
                return (ineigh < 0 ? empty_ : data_[ineigh]); 
            }

            GridData<T> & operator=(const GridData<T> & other) {
                assert(grid_ == other.grid_);
                data_ = other.data_;
                return *this;
            }

            // always defined
            void fill(const T &val) { std::fill(data_.begin(), data_.end(), val); }
            void zero() { fill(T()); }

            // defined only if T has a 'clear' method
            void clear() { for (T & t : data_) t.clear(); }
            
        private:
            const Grid *   grid_;
            std::vector<T> data_;
            const T        empty_;
    };
    typedef GridData<float> EtGrid;

    struct PreCluster {
        PreCluster() : ptLocalMax(0), ptOverNeighLocalMaxSum(0) {}
        float ptLocalMax; // pt if it's a local max, zero otherwise
        float ptOverNeighLocalMaxSum; // pt / (sum of ptLocalMax of neighbours); zero if no neighbours
        void clear() { ptLocalMax = ptOverNeighLocalMaxSum = 0; }
    };
    typedef GridData<PreCluster> PreClusterGrid;

    struct Cluster {
        Cluster() : et(0), eta(0), phi(0) {}
        float et, eta, phi;
        void clear() { et = eta = phi = 0; }
    };
    typedef GridData<Cluster> ClusterGrid;

    struct CombinedCluster : public Cluster {
        float ecal_et, hcal_et;
        void clear() { et = ecal_et = hcal_et = eta = phi = 0; }
    };
    typedef GridData<CombinedCluster> CombinedClusterGrid;

    const Grid * getGrid(const std::string & type) ;

    class SingleCaloClusterer {
        public:
            SingleCaloClusterer(const edm::ParameterSet &pset) ;
            ~SingleCaloClusterer() ;
            void clear() { rawet_.zero(); }
            void add(const reco::Candidate &c) { add(c.pt(), c.eta(), c.phi()); }
            void add(float pt, float eta, float phi) { 
                rawet_(eta, phi) += pt; 
            }
            void run() ; 

            /// possibly grow clusters by adding unclustered energy on the sides
            //  note: there can be some double-counting as the same unclustered energy can go into more clusters
            void grow() ; 

            const EtGrid      & raw()      const { return rawet_; }
            const ClusterGrid & clusters() const { return cluster_; }

            // for the moment, generic interface that takes a cluster and returns the corrected pt
            template<typename Corrector>
            void correct(const Corrector & corrector) {
                for (unsigned int i = 0, ncells = grid_->size(); i < ncells; ++i) {
                    if (cluster_[i].et > 0) {
                        cluster_[i].et = corrector(cluster_[i]);
                    }
                }
            }

            std::unique_ptr<l1t::PFClusterCollection> fetch(float ptMin=0.) const ;
            std::unique_ptr<l1t::PFClusterCollection> fetchCells(float ptMin, bool unclusteredOnly) const ;

        private:
            enum EnergyShareAlgo { Fractions, /* each local maximum neighbour takes a share proportional to its value */
                                   None,      /* each local maximum neighbour takes all the value (double counting!) */
                                   Greedy,    /* assing cell to the highest local maximum neighbour */
                                   Crude };   /* if there's more than one local maximum neighbour, they all take half of the value (no fp division) */
            const Grid * grid_;
            EtGrid         rawet_, unclustered_;
            PreClusterGrid precluster_;
            ClusterGrid    cluster_;
            float zsEt_, seedEt_, minClusterEt_, minEtToGrow_;
            EnergyShareAlgo energyShareAlgo_;
            bool  energyWeightedPosition_; // do the energy-weighted cluster position instead of the cell center
    };

    class SimpleCaloLinker {
        public:
            SimpleCaloLinker(const edm::ParameterSet &pset, const SingleCaloClusterer & ecal,  const SingleCaloClusterer & hcal) ;
            ~SimpleCaloLinker() ;
            void run() ; 
            const CombinedClusterGrid & clusters() const { return cluster_; }
            
            // for the moment, generic interface that takes a cluster and returns the corrected pt
            template<typename Corrector>
            void correct(const Corrector & corrector) {
                for (unsigned int i = 0, ncells = grid_->size(); i < ncells; ++i) {
                    if (cluster_[i].et > 0) {
                        cluster_[i].et = corrector(cluster_[i]);
                    }
                }
            }

            std::unique_ptr<l1t::PFClusterCollection> fetch() const ;
        private:
            const Grid * grid_;
            const SingleCaloClusterer & ecal_, & hcal_;
            PreClusterGrid ecalToHCal_;
            CombinedClusterGrid cluster_;
            float hoeCut_, minPhotonEt_, minHadronRawEt_, minHadronEt_;
    };

} // namespace 

#endif
