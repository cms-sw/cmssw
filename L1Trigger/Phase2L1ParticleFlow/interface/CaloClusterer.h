#ifndef L1Trigger_Phase2L1ParticleFlow_CALOCLUSTERER_H
#define L1Trigger_Phase2L1ParticleFlow_CALOCLUSTERER_H
/** 
 * Classes for calorimetric re-clustering
 * */

// fwd declarations
namespace edm {
  class ParameterSet;
}

// real includes
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

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
    Grid(unsigned int size)
        : ncells_(size),
          eta_(size),
          etaWidth_(size),
          phi_(size),
          phiWidth_(size),
          ieta_(size),
          iphi_(size),
          neighbours_(size) {}
    unsigned int ncells_;
    std::vector<float> eta_, etaWidth_, phi_, phiWidth_;
    std::vector<int> ieta_, iphi_;
    std::vector<std::array<int, 8>> neighbours_;  // indices of the neigbours, -1 = none
  };

  class Phase1GridBase : public Grid {
  public:
    Phase1GridBase(int nEta, int nPhi, int ietaCoarse, int ietaVeryCoarse, const float *towerEtas);

    int find_cell(float eta, float phi) const override;
    int ifind_cell(int ieta, int iphi) const { return cell_map_[(ieta + nEta_) + 2 * nEta_ * (iphi - 1)]; }

  protected:
    const int nEta_, nPhi_, ietaCoarse_, ietaVeryCoarse_;
    const float *towerEtas_;
    std::vector<int> cell_map_;
    // valid ieta, iphi (does not check for outside bounds, only for non-existence of ieta=0, iphi=0, and coarser towers at high eta)
    bool valid_ieta_iphi(int ieta, int iphi) const {
      if (ieta == 0 || iphi == 0)
        return false;
      if (std::abs(ieta) >= ietaVeryCoarse_ && (iphi % 4 != 1))
        return false;
      if (std::abs(ieta) >= ietaCoarse_ && (iphi % 2 != 1))
        return false;
      return true;
    }
    // move by +/-1 around a cell; return icell or -1 if not available
    int imove(int ieta, int iphi, int deta, int dphi);
  };

  class Phase1Grid : public Phase1GridBase {
  public:
    Phase1Grid()
        : Phase1GridBase(phase1_nEta_, phase1_nPhi_, phase1_ietaCoarse_, phase1_ietaVeryCoarse_, phase1_towerEtas_) {}

  protected:
    static const int phase1_nEta_ = 41, phase1_nPhi_ = 72, phase1_ietaCoarse_ = 29, phase1_ietaVeryCoarse_ = 40;
    static const float phase1_towerEtas_[phase1_nEta_];
  };
  class Phase2Grid : public Phase1GridBase {
  public:
    Phase2Grid()
        : Phase1GridBase(phase2_nEta_, phase2_nPhi_, phase2_ietaCoarse_, phase2_ietaVeryCoarse_, phase2_towerEtas_) {}

  protected:
    static const int phase2_nEta_ = 48, phase2_nPhi_ = 72, phase2_ietaCoarse_ = 36, phase2_ietaVeryCoarse_ = 47;
    static const float phase2_towerEtas_[phase2_nEta_];
  };

  template <typename T>
  class GridData {
  public:
    GridData() : grid_(nullptr), data_(), empty_() {}
    GridData(const Grid &grid) : grid_(&grid), data_(grid.size()), empty_() {}

    T &operator()(float eta, float phi) { return data_[grid_->find_cell(eta, phi)]; }
    const T &operator()(float eta, float phi) const { return data_[grid_->find_cell(eta, phi)]; }

    float eta(float eta, float phi) const { return grid().eta(grid_->find_cell(eta, phi)); }
    float phi(float eta, float phi) const { return grid().phi(grid_->find_cell(eta, phi)); }

    const Grid &grid() const { return *grid_; }

    unsigned int size() const { return data_.size(); }

    float eta(int icell) const { return grid().eta(icell); }
    float phi(int icell) const { return grid().phi(icell); }
    int ieta(int icell) const { return grid().ieta(icell); }
    int iphi(int icell) const { return grid().iphi(icell); }

    T &operator[](int icell) { return data_[icell]; }
    const T &operator[](int icell) const { return data_[icell]; }

    const T &neigh(int icell, unsigned int idx) const {
      int ineigh = grid_->neighbour(icell, idx);
      return (ineigh < 0 ? empty_ : data_[ineigh]);
    }

    GridData<T> &operator=(const GridData<T> &other) {
      assert(grid_ == other.grid_);
      data_ = other.data_;
      return *this;
    }
    GridData<T> &operator+=(const GridData<T> &other) {
      assert(grid_ == other.grid_);
      for (unsigned int i = 0, n = data_.size(); i < n; ++i) {
        data_[i] += other.data_[i];
      }
      return *this;
    }

    // always defined
    void fill(const T &val) { std::fill(data_.begin(), data_.end(), val); }
    void zero() { fill(T()); }

    // defined only if T has a 'clear' method
    void clear() {
      for (T &t : data_)
        t.clear();
    }

  private:
    const Grid *grid_;
    std::vector<T> data_;
    const T empty_;
  };
  typedef GridData<float> EtaPhiCenterGrid;
  typedef GridData<float> EtGrid;
  typedef GridData<int> IndexGrid;

  struct PreCluster {
    PreCluster() : ptLocalMax(0), ptOverNeighLocalMaxSum(0) {}
    float ptLocalMax;              // pt if it's a local max, zero otherwise
    float ptOverNeighLocalMaxSum;  // pt / (sum of ptLocalMax of neighbours); zero if no neighbours
    void clear() { ptLocalMax = ptOverNeighLocalMaxSum = 0; }
  };
  typedef GridData<PreCluster> PreClusterGrid;

  struct Cluster {
    Cluster() : et(0), eta(0), phi(0) {}
    float et, eta, phi;
    std::vector<std::pair<int, float>> constituents;
    void clear() {
      et = eta = phi = 0;
      constituents.clear();
    }
  };

  struct CombinedCluster : public Cluster {
    float ecal_et, hcal_et;
    float ecal_eta, ecal_phi;
    void clear() {
      Cluster::clear();
      ecal_et = hcal_et = 0;
      ecal_eta = ecal_phi = 0;
    }
  };

  const Grid *getGrid(const std::string &type);

  class SingleCaloClusterer {
  public:
    SingleCaloClusterer(const edm::ParameterSet &pset);
    ~SingleCaloClusterer();
    void clear();
    void add(const reco::Candidate &c, bool updateEtaPhi = false) { add(c.pt(), c.eta(), c.phi(), updateEtaPhi); }
    void add(float pt, float eta, float phi, bool updateEtaPhi = false) { 
      rawet_(eta, phi) += pt; 
      if (updateEtaPhi) {
        float newet = rawet_(eta, phi);
        float prevw = (newet-pt)/newet;
        float nextw = pt/newet;
        eta_center_(eta, phi) = eta_center_(eta, phi)*prevw + eta*nextw;
        phi_center_(eta, phi) = phi_center_(eta, phi)*prevw + phi*nextw;
      }
    }
    void run();

    /// possibly grow clusters by adding unclustered energy on the sides
    //  note: there can be some double-counting as the same unclustered energy can go into more clusters
    void grow();

    const EtGrid &raw() const { return rawet_; }
    const EtaPhiCenterGrid &etaCenter() const { return eta_center_; }
    const EtaPhiCenterGrid &phiCenter() const { return phi_center_; }
    const IndexGrid &indexGrid() const { return clusterIndex_; }
    const std::vector<Cluster> &clusters() const { return clusters_; }
    const Cluster &cluster(int i) const {
      return (i == -1 || clusterIndex_[i] == -1) ? nullCluster_ : clusters_[clusterIndex_[i]];
    }

    /// non-const access to the energy: be careful to use it only before 'run()'
    EtGrid &raw() { return rawet_; }
    EtaPhiCenterGrid &etaCenter() { return eta_center_; }
    EtaPhiCenterGrid &phiCenter() { return phi_center_; }

    // for the moment, generic interface that takes a cluster and returns the corrected pt
    template <typename Corrector>
    void correct(const Corrector &corrector) {
      for (Cluster &c : clusters_) {
        c.et = corrector(c);
      }
    }

    std::unique_ptr<l1t::PFClusterCollection> fetchCells(bool unclusteredOnly = false, float ptMin = 0.) const;

    std::unique_ptr<l1t::PFClusterCollection> fetch(float ptMin = 0.) const;
    std::unique_ptr<l1t::PFClusterCollection> fetch(const edm::OrphanHandle<l1t::PFClusterCollection> &cells,
                                                    float ptMin = 0.) const;

  private:
    enum class EnergyShareAlgo {
      Fractions, /* each local maximum neighbour takes a share proportional to its value */
      None,      /* each local maximum neighbour takes all the value (double counting!) */
      Greedy,    /* assing cell to the highest local maximum neighbour */
      Crude
    }; /* if there's more than one local maximum neighbour, they all take half of the value (no fp division) */
    const Grid *grid_;
    EtGrid rawet_, unclustered_;
    EtaPhiCenterGrid eta_center_;
    EtaPhiCenterGrid phi_center_;
    PreClusterGrid precluster_;
    IndexGrid clusterIndex_, cellKey_;
    std::vector<Cluster> clusters_;
    const Cluster nullCluster_;
    float zsEt_, seedEt_, minClusterEt_, minEtToGrow_;
    EnergyShareAlgo energyShareAlgo_;
    bool energyWeightedPosition_;  // do the energy-weighted cluster position instead of the cell center
  };

  class SimpleCaloLinkerBase {
  public:
    SimpleCaloLinkerBase(const edm::ParameterSet &pset,
                         const SingleCaloClusterer &ecal,
                         const SingleCaloClusterer &hcal);
    virtual ~SimpleCaloLinkerBase();
    virtual void clear() { clearBase(); }
    virtual void run() = 0;
    void clearBase() {
      clusters_.clear();
      clusterIndex_.fill(-1);
    }

    // for the moment, generic interface that takes a cluster and returns the corrected pt
    template <typename Corrector>
    void correct(const Corrector &corrector) {
      for (CombinedCluster &c : clusters_) {
        c.et = corrector(c);
      }
    }

    std::unique_ptr<l1t::PFClusterCollection> fetch() const;
    std::unique_ptr<l1t::PFClusterCollection> fetch(const edm::OrphanHandle<l1t::PFClusterCollection> &ecal,
                                                    const edm::OrphanHandle<l1t::PFClusterCollection> &hcal) const;

  protected:
    const Grid *grid_;
    const SingleCaloClusterer &ecal_, &hcal_;
    IndexGrid clusterIndex_;
    std::vector<CombinedCluster> clusters_;
    float hoeCut_, minPhotonEt_, minHadronRawEt_, minHadronEt_;
    bool noEmInHGC_;
  };

  class SimpleCaloLinker : public SimpleCaloLinkerBase {
  public:
    SimpleCaloLinker(const edm::ParameterSet &pset, const SingleCaloClusterer &ecal, const SingleCaloClusterer &hcal);
    ~SimpleCaloLinker() override;
    void clear() override;
    void run() override;

  protected:
    PreClusterGrid ecalToHCal_;
  };
  class FlatCaloLinker : public SimpleCaloLinkerBase {
  public:
    FlatCaloLinker(const edm::ParameterSet &pset, const SingleCaloClusterer &ecal, const SingleCaloClusterer &hcal);
    ~FlatCaloLinker() override;
    void clear() override;
    void run() override;

  protected:
    SingleCaloClusterer combClusterer_;
  };

  class CombinedCaloLinker : public SimpleCaloLinkerBase {
  public:
    CombinedCaloLinker(const edm::ParameterSet &pset, const SingleCaloClusterer &ecal, const SingleCaloClusterer &hcal);
    ~CombinedCaloLinker() override;
    void clear() override;
    void run() override;

  protected:
    SingleCaloClusterer combClusterer_;
  };

  // makes a calo linker (pointer will be owned by the callee)
  std::unique_ptr<SimpleCaloLinkerBase> makeCaloLinker(const edm::ParameterSet &pset,
                                                       const SingleCaloClusterer &ecal,
                                                       const SingleCaloClusterer &hcal);

}  // namespace l1tpf_calo

#endif
