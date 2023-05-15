#include "L1Trigger/Phase2L1ParticleFlow/interface/CaloClusterer.h"

#include <cassert>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

const float l1tpf_calo::Phase1Grid::phase1_towerEtas_[l1tpf_calo::Phase1Grid::phase1_nEta_] = {
    0,     0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870, 0.957, 1.044, 1.131,
    1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830, 1.930, 2.043, 2.172, 2.322, 2.5,   2.650,
    2.853, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};
const float l1tpf_calo::Phase2Grid::phase2_towerEtas_[l1tpf_calo::Phase2Grid::phase2_nEta_] = {
    0,     0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870, 0.957, 1.044, 1.131, 1.218, 1.305,
    1.392, 1.479, 1.564, 1.648, 1.732, 1.817, 1.901, 1.986, 2.071, 2.155, 2.240, 2.324, 2.409, 2.493, 2.577, 2.662,
    2.747, 2.831, 2.915, 3.0,   3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};

l1tpf_calo::Phase1GridBase::Phase1GridBase(
    int nEta, int nPhi, int ietaCoarse, int ietaVeryCoarse, const float *towerEtas)
    : Grid(2 * ((ietaCoarse - 1) * nPhi + (ietaVeryCoarse - ietaCoarse) * (nPhi / 2) +
                (nEta - ietaVeryCoarse + 1) * (nPhi / 4))),
      nEta_(nEta),
      nPhi_(nPhi),
      ietaCoarse_(ietaCoarse),
      ietaVeryCoarse_(ietaVeryCoarse),
      towerEtas_(towerEtas),
      cell_map_(2 * nEta * nPhi, -1) {
  int icell = 0;
  for (int ie = -nEta_; ie <= nEta_; ++ie) {
    int absie = std::abs(ie);
    for (int iph = 1; iph <= nPhi_; ++iph) {
      if (!valid_ieta_iphi(ie, iph))
        continue;
      ieta_[icell] = ie;
      iphi_[icell] = iph;
      float etaLo = (absie < nEta_ ? towerEtas_[absie - 1] : towerEtas_[absie - 2]);
      float etaHi = (absie < nEta_ ? towerEtas_[absie] : towerEtas_[absie - 1]);
      eta_[icell] = (ie > 0 ? 0.5 : -0.5) * (etaLo + etaHi);
      etaWidth_[icell] = (etaHi - etaLo);
      phiWidth_[icell] = 2 * M_PI / nPhi_;
      if (absie >= ietaVeryCoarse_)
        phiWidth_[icell] *= 4;
      else if (absie >= ietaCoarse_)
        phiWidth_[icell] *= 2;
      phi_[icell] = (iph - 1) * 2 * M_PI / nPhi_ + 0.5 * phiWidth_[icell];
      if (phi_[icell] > M_PI)
        phi_[icell] -= 2 * M_PI;
      std::fill(neighbours_[icell].begin(), neighbours_[icell].end(), -1);
      cell_map_[(ie + nEta_) + 2 * nEta_ * (iph - 1)] = icell;
      icell++;
    }
  }
  assert(unsigned(icell) == ncells_);
  // now link the cells
  for (icell = 0; icell < int(ncells_); ++icell) {
    int ie = ieta_[icell], iph = iphi_[icell];
    int ineigh = 0;
    for (int deta = -1; deta <= +1; ++deta) {
      for (int dphi = -1; dphi <= +1; ++dphi) {
        if (deta == 0 && dphi == 0)
          continue;
        neighbours_[icell][ineigh++] = imove(ie, iph, deta, dphi);
      }
    }
  }
  //// consistency check 1: check that find_cell works
  //// uncomment to check that there's no holes in the grid
  //for (float teta = 0; teta <= 5.0; teta += 0.02) {
  //    for (float tphi = -M_PI; tphi <= M_PI; tphi += 0.02) {
  //        find_cell(+teta, tphi);
  //        find_cell(-teta, tphi);
  //    }
  //}
}

int l1tpf_calo::Phase1GridBase::find_cell(float eta, float phi) const {
  int ieta =
      (eta != 0) ? std::distance(towerEtas_, std::lower_bound(towerEtas_, towerEtas_ + nEta_, std::abs(eta))) : 1;
  if (ieta == nEta_)
    return -1;  // outside bounds
  assert(ieta > 0 && ieta < nEta_);
  if (ieta > nEta_)
    ieta = nEta_;
  if (eta < 0)
    ieta = -ieta;
  phi = reco::reduceRange(phi);  // [-PI, PI]
  if (phi < 0)                   // then bring to [0, 2*PI]
    phi += 2 * M_PI;
  int iphi = std::floor(phi * nPhi_ / (2 * M_PI));
  if (phi >= 2 * M_PI)
    iphi = nPhi_ - 1;  // fix corner case due to roundings etc
  assert(iphi < nPhi_);
  if (std::abs(ieta) >= ietaVeryCoarse_)
    iphi -= (iphi % 4);
  else if (std::abs(ieta) >= ietaCoarse_)
    iphi -= (iphi % 2);
  iphi += 1;
  //// uncomment to check validity of derived coordinates
  //if (!valid_ieta_iphi(ieta,iphi)) {
  //    printf("Error in finding cell for eta %+7.4f phi %+7.4f, got ieta = %+3d iphi %2d which is not valid\n",
  //        eta, phi, ieta, iphi);
  //}
  assert(valid_ieta_iphi(ieta, iphi));
  int icell = ifind_cell(ieta, iphi);
  assert(icell != -1);

  //// uncomment to check that the point is really in the cell
  //if (std::abs(eta - eta_[icell]) > 0.501*etaWidth_[icell] || std::abs(deltaPhi(phi, phi_[icell])) > 0.501*phiWidth_[icell]) {
  //    printf("Mismatch in finding cell for eta %+7.4f phi %+7.4f, got ieta = %+3d iphi %2d which has eta %+7.4f +- %.4f phi %+7.4f +- %.4f ; deta = %+7.4f dphi = %+7.4f\n",
  //        eta, phi, ieta, iphi, eta_[icell], etaWidth_[icell], phi_[icell], phiWidth_[icell], eta - eta_[icell], deltaPhi(phi, phi_[icell]));
  //}
  //assert(std::abs(eta - eta_[icell]) <= 0.5*etaWidth_[icell]);
  //assert(std::abs(deltaPhi(phi, phi_[icell])) <= 0.5*phiWidth_[icell]);
  return icell;
}

int l1tpf_calo::Phase1GridBase::imove(int ieta, int iphi, int deta, int dphi) {
  int ie = ieta, iph = iphi;
  switch (deta) {
    case -1:
      ie = (ie == -nEta_ ? 0 : (ie == +1 ? -1 : ie - 1));
      break;
    case +1:
      ie = (ie == +nEta_ ? 0 : (ie == -1 ? +1 : ie + 1));
      break;
    case 0:
      break;
    default:
      assert(false);
  };
  if (ie == 0)
    return -1;
  switch (dphi) {
    case -1:
      iph = (iph == 1 ? nPhi_ : iph - 1);
      break;
    case +1:
      iph = (iph == nPhi_ ? 1 : iph + 1);
      break;
    case 0:
      break;
    default:
      assert(false);
  };
  if (!valid_ieta_iphi(ie, iph))
    return -1;
  int icell = ifind_cell(ie, iph);
  assert(!(ie == ieta && iph == iphi));
  assert(icell != -1);
  assert(icell != ifind_cell(ieta, iphi));
  return icell;
}

const l1tpf_calo::Grid *l1tpf_calo::getGrid(const std::string &type) {
  static const Phase1Grid _phase1Grid;
  static const Phase2Grid _phase2Grid;
  if (type == "phase1")
    return &_phase1Grid;
  else if (type == "phase2")
    return &_phase2Grid;
  else
    throw cms::Exception("Configuration") << "Unsupported grid type '" << type << "'\n";
}

l1tpf_calo::SingleCaloClusterer::SingleCaloClusterer(const edm::ParameterSet &pset)
    : grid_(getGrid(pset.getParameter<std::string>("grid"))),
      rawet_(*grid_),
      unclustered_(*grid_),
      eta_center_(*grid_),
      phi_center_(*grid_),
      precluster_(*grid_),
      clusterIndex_(*grid_),
      cellKey_(*grid_),
      preciseEtaPhi_(pset.existsAs<bool>("usePreciseEtaPhi") ? pset.getParameter<bool>("usePreciseEtaPhi") : false),
      etaBounds_(pset.getParameter<std::vector<double>>("etaBounds")),
      phiBounds_(pset.getParameter<std::vector<double>>("phiBounds")),
      maxClustersEtaPhi_(pset.getParameter<std::vector<unsigned int>>("maxClustersEtaPhi")),
      clusters_(),
      nullCluster_(),
      zsEt_(pset.getParameter<double>("zsEt")),
      seedEt_(pset.getParameter<double>("seedEt")),
      minClusterEt_(pset.getParameter<double>("minClusterEt")),
      minEtToGrow_(pset.existsAs<double>("minEtToGrow") ? pset.getParameter<double>("minEtToGrow") : -1),
      energyWeightedPosition_(pset.getParameter<bool>("energyWeightedPosition")) {
  std::string energyShareAlgo = pset.getParameter<std::string>("energyShareAlgo");
  if (energyShareAlgo == "fractions")
    energyShareAlgo_ = EnergyShareAlgo::Fractions;
  else if (energyShareAlgo == "none")
    energyShareAlgo_ = EnergyShareAlgo::None;
  else if (energyShareAlgo == "greedy")
    energyShareAlgo_ = EnergyShareAlgo::Greedy;
  else if (energyShareAlgo == "crude")
    energyShareAlgo_ = EnergyShareAlgo::Crude;
  else
    throw cms::Exception("Configuration") << "Unsupported energyShareAlgo '" << energyShareAlgo << "'\n";
  if (pset.existsAs<std::vector<int>>("neighborCells")) {
    neighborCells_ = pset.getParameter<std::vector<int>>(
        "neighborCells");  //anything other than 3x3 is incompatible with grow() I think...
  } else {
    neighborCells_ = std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7});  //default to 3x3
    //  in relative eta,phi: 5 = (+1, 0), 6 = (+1, 0), 7 = (+1,+1)
    //                       3 = ( 0,-1),              4 = ( 0,+1),
    //                       0 = (-1,-1), 1 = (-1, 0), 2 = (-1,+1),
  }
  if ((etaBounds_.size() - 1) * (phiBounds_.size() - 1) != maxClustersEtaPhi_.size()) {
    throw cms::Exception("Configuration")
        << "Size mismatch between eta/phi bounds and max clusters: " << (etaBounds_.size() - 1) << " x "
        << (phiBounds_.size() - 1) << " != " << maxClustersEtaPhi_.size() << "\n";
  }
  if (!std::is_sorted(etaBounds_.begin(), etaBounds_.end())) {
    throw cms::Exception("Configuration") << "etaBounds is not sorted\n";
  }
  if (!std::is_sorted(phiBounds_.begin(), phiBounds_.end())) {
    throw cms::Exception("Configuration") << "phiBounds is not sorted\n";
  }
}

l1tpf_calo::SingleCaloClusterer::~SingleCaloClusterer() {}

void l1tpf_calo::SingleCaloClusterer::clear() {
  rawet_.zero();
  eta_center_.zero();
  phi_center_.zero();
  clusters_.clear();
  clusterIndex_.fill(-1);
}

void l1tpf_calo::SingleCaloClusterer::run() {
  unsigned int i, ncells = grid_->size();

  // kill zeros. count non-zeros, for linking later
  cellKey_.fill(-1);
  int key = 0;
  for (i = 0; i < ncells; ++i) {
    if (rawet_[i] < zsEt_) {
      rawet_[i] = 0;
    } else {
      cellKey_[i] = key++;
    }
  }

  precluster_.clear();
  // pre-cluster step 1: at each cell, set the value equal to itself if it's a local maxima, zero otherwise
  // can be done in parallel on all cells
  for (i = 0; i < ncells; ++i) {
    if (rawet_[i] > seedEt_) {
      precluster_[i].ptLocalMax = rawet_[i];
      //// uncommment code below for debugging the clustering
      //printf("   candidate precluster pt %7.2f at %4d (ieta %+3d iphi %2d)\n",  rawet_[i], i, grid_->ieta(i), grid_->iphi(i));
      for (const auto &ineigh : neighborCells_) {
        if (ineigh >= 4)
          continue;
        if (rawet_.neigh(i, ineigh) > rawet_[i])
          precluster_[i].ptLocalMax = 0;
        //// uncommment code below for debugging the clustering
        //int ncell = grid_->neighbour(i,ineigh);
        //if (ncell == -1) printf("   \t neigh %d is null\n", ineigh);
        //else printf("   \t neigh %d at %4d (ieta %+3d iphi %2d) has pt %7.2f: comparison %1d \n", ineigh, ncell, grid_->ieta(ncell), grid_->iphi(ncell), rawet_[ncell], precluster_[i].ptLocalMax > 0);
      }
      for (const auto &ineigh : neighborCells_) {
        if (ineigh < 4)
          continue;
        if (rawet_.neigh(i, ineigh) >= rawet_[i])
          precluster_[i].ptLocalMax = 0;
        //// uncommment code below for debugging the clustering
        //int ncell = grid_->neighbour(i,ineigh);
        //if (ncell == -1) printf("   \t neigh %d is null\n", ineigh);
        //else printf("   \t neigh %d at %4d (ieta %+3d iphi %2d) has pt %7.2f: comparison %1d \n", ineigh, ncell, grid_->ieta(ncell), grid_->iphi(ncell), rawet_[ncell], precluster_[i].ptLocalMax > 0);
      }
    }
  }
  // pre-cluster step 2: compute information from neighbouring local max, for energy sharing purposes
  for (i = 0; i < ncells; ++i) {
    if (precluster_[i].ptLocalMax == 0) {
      switch (energyShareAlgo_) {
        case EnergyShareAlgo::Fractions: {
          float tot = 0;
          for (const auto &ineigh : neighborCells_) {
            tot += precluster_.neigh(i, ineigh).ptLocalMax;
          }
          precluster_[i].ptOverNeighLocalMaxSum = tot ? rawet_[i] / tot : 0;
        } break;
        case EnergyShareAlgo::None:
          precluster_[i].ptOverNeighLocalMaxSum = rawet_[i];
          break;
        case EnergyShareAlgo::Greedy: {
          float maxet = 0;
          for (const auto &ineigh : neighborCells_) {
            maxet = std::max(maxet, precluster_.neigh(i, ineigh).ptLocalMax);
          }
          precluster_[i].ptOverNeighLocalMaxSum = maxet;
        } break;
        case EnergyShareAlgo::Crude: {
          int number = 0;
          for (const auto &ineigh : neighborCells_) {
            number += (precluster_.neigh(i, ineigh).ptLocalMax > 0);
          }
          precluster_[i].ptOverNeighLocalMaxSum = (number > 1 ? 0.5 : 1.0) * rawet_[i];
        } break;
      }
    }
  }

  clusterIndex_.fill(-1);
  clusters_.clear();
  unclustered_ = rawet_;
  // cluster: at each localMax cell, take itself plus the weighted contributions of the neighbours
  Cluster cluster;
  for (i = 0; i < ncells; ++i) {
    if (precluster_[i].ptLocalMax > 0) {
      float myet = rawet_[i];
      float tot = myet;
      float avg_eta = 0;
      float avg_phi = 0;
      cluster.clear();
      cluster.constituents.emplace_back(i, 1.0);
      for (const auto &ineigh : neighborCells_) {
        int ineighcell = grid_->neighbour(i, ineigh);
        if (ineighcell == -1)
          continue;  // skip dummy cells
        float fracet = 0;
        switch (energyShareAlgo_) {
          case EnergyShareAlgo::Fractions:
            fracet = myet * precluster_.neigh(i, ineigh).ptOverNeighLocalMaxSum;
            break;
          case EnergyShareAlgo::None:
            fracet = precluster_.neigh(i, ineigh).ptOverNeighLocalMaxSum;
            break;
          case EnergyShareAlgo::Greedy:
            fracet = (myet == precluster_.neigh(i, ineigh).ptOverNeighLocalMaxSum ? rawet_.neigh(i, ineigh) : 0);
            break;
          case EnergyShareAlgo::Crude:
            fracet = precluster_.neigh(i, ineigh).ptOverNeighLocalMaxSum;
            break;
        }
        if (fracet == 0)
          continue;
        tot += fracet;
        cluster.constituents.emplace_back(ineighcell, fracet / rawet_.neigh(i, ineigh));
        if (energyWeightedPosition_) {
          avg_eta += fracet * (grid_->eta(ineighcell) - grid_->eta(i));
          avg_phi += fracet * deltaPhi(grid_->phi(ineighcell), grid_->phi(i));
        }
      }
      if (tot > minClusterEt_) {
        cluster.et = tot;
        unclustered_[i] = 0;
        for (const auto &ineigh : neighborCells_) {
          int ineighcell = grid_->neighbour(i, ineigh);
          if (ineighcell == -1)
            continue;  // skip dummy cells
          unclustered_[ineighcell] = 0;
        }
        if (energyWeightedPosition_) {
          cluster.eta = grid_->eta(i) + avg_eta / tot;
          cluster.phi = grid_->phi(i) + avg_phi / tot;
          // wrap around phi
          cluster.phi = reco::reduceRange(cluster.phi);
        } else {
          cluster.eta = grid_->eta(i);
          cluster.phi = grid_->phi(i);
        }
        clusterIndex_[i] = clusters_.size();
        clusters_.push_back(cluster);
      }
    }
  }
  if (minEtToGrow_ > 0)
    grow();
}

void l1tpf_calo::SingleCaloClusterer::grow() {
  int selneighs[4] = {1, 3, 4, 6};  // -eta, -phi, +phi, +eta
  std::vector<int> toreset;
  for (Cluster &cluster : clusters_) {
    if (cluster.et > minEtToGrow_) {
      int i = cluster.constituents.front().first;
      for (int side = 0; side < 4; ++side) {
        int neigh = grid_->neighbour(i, selneighs[side]);
        if (neigh == -1)
          continue;
        for (int in = 0; in < 8; ++in) {
          int n2 = grid_->neighbour(neigh, in);
          if (n2 == -1)
            continue;
          cluster.et += unclustered_[n2];
          if (unclustered_[n2]) {
            cluster.constituents.emplace_back(n2, 1.0);
            toreset.push_back(n2);
          }
        }
      }
    }
  }
  for (int i : toreset)
    unclustered_[i] = 0;
}

std::unique_ptr<l1t::PFClusterCollection> l1tpf_calo::SingleCaloClusterer::fetchCells(bool unclusteredOnly,
                                                                                      float ptMin) const {
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  const EtGrid &src = (unclusteredOnly ? unclustered_ : rawet_);
  const EtaPhiCenterGrid &eta_shift = eta_center_;
  const EtaPhiCenterGrid &phi_shift = phi_center_;
  l1tpf_calo::GridSelector selector = l1tpf_calo::GridSelector(etaBounds_, phiBounds_, maxClustersEtaPhi_);
  int totalClusters = 0;
  for (unsigned int i = 0, ncells = grid_->size(); i < ncells; ++i) {
    if (src[i] <= ptMin)
      continue;
    if ((unclusteredOnly == false) && (ptMin == 0)) {
      assert(cellKey_[i] == totalClusters);
    }
    totalClusters++;
    selector.fill(src[i], grid_->eta(i), grid_->phi(i), i);
  }
  std::vector<unsigned int> indices = selector.returnSorted();
  for (unsigned int ii = 0; ii < indices.size(); ii++) {
    unsigned int theIndex = indices[ii];
    ret->emplace_back(
        src[theIndex], grid_->eta(theIndex) + eta_shift[theIndex], grid_->phi(theIndex) + phi_shift[theIndex]);
    ret->back().setHwEta(grid_->ieta(theIndex));
    ret->back().setHwPhi(grid_->iphi(theIndex));
  }
  return ret;
}

std::unique_ptr<l1t::PFClusterCollection> l1tpf_calo::SingleCaloClusterer::fetch(float ptMin) const {
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  for (const Cluster &cluster : clusters_) {
    if (cluster.et > ptMin) {
      ret->emplace_back(cluster.et, cluster.eta, cluster.phi);
    }
  }
  return ret;
}

std::unique_ptr<l1t::PFClusterCollection> l1tpf_calo::SingleCaloClusterer::fetch(
    const edm::OrphanHandle<l1t::PFClusterCollection> &cells, float ptMin) const {
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  for (const Cluster &cluster : clusters_) {
    if (cluster.et > ptMin) {
      ret->emplace_back(cluster.et, cluster.eta, cluster.phi);
      for (const auto &pair : cluster.constituents) {
        edm::Ptr<l1t::PFCluster> ref(cells, cellKey_[pair.first]);
        ret->back().addConstituent(ref, pair.second);
      }
    }
  }
  return ret;
}

l1tpf_calo::SimpleCaloLinkerBase::SimpleCaloLinkerBase(const edm::ParameterSet &pset,
                                                       const SingleCaloClusterer &ecal,
                                                       const SingleCaloClusterer &hcal)
    : grid_(getGrid(pset.getParameter<std::string>("grid"))),
      ecal_(ecal),
      hcal_(hcal),
      clusterIndex_(*grid_),
      clusters_(),
      etaBounds_(pset.getParameter<std::vector<double>>("etaBounds")),
      phiBounds_(pset.getParameter<std::vector<double>>("phiBounds")),
      maxClustersEtaPhi_(pset.getParameter<std::vector<unsigned int>>("maxClustersEtaPhi")),
      hoeCut_(pset.getParameter<double>("hoeCut")),
      minPhotonEt_(pset.getParameter<double>("minPhotonEt")),
      minHadronRawEt_(pset.getParameter<double>("minHadronRawEt")),
      minHadronEt_(pset.getParameter<double>("minHadronEt")),
      noEmInHGC_(pset.getParameter<bool>("noEmInHGC")) {
  if (grid_ != &ecal.raw().grid())
    throw cms::Exception("LogicError", "Inconsistent grid between ecal and linker\n");
  if (grid_ != &hcal.raw().grid())
    throw cms::Exception("LogicError", "Inconsistent grid between hcal and linker\n");
  if ((etaBounds_.size() - 1) * (phiBounds_.size() - 1) != maxClustersEtaPhi_.size()) {
    throw cms::Exception("Configuration")
        << "Size mismatch between eta/phi bounds and max clusters: " << (etaBounds_.size() - 1) << " x "
        << (phiBounds_.size() - 1) << " != " << maxClustersEtaPhi_.size() << "\n";
  }
  if (!std::is_sorted(etaBounds_.begin(), etaBounds_.end())) {
    throw cms::Exception("Configuration") << "etaBounds is not sorted\n";
  }
  if (!std::is_sorted(phiBounds_.begin(), phiBounds_.end())) {
    throw cms::Exception("Configuration") << "phiBounds is not sorted\n";
  }
}

l1tpf_calo::SimpleCaloLinkerBase::~SimpleCaloLinkerBase() {}

std::unique_ptr<l1t::PFClusterCollection> l1tpf_calo::SimpleCaloLinkerBase::fetch() const {
  edm::OrphanHandle<l1t::PFClusterCollection> ecal, hcal;
  return fetch(ecal, hcal);
}

std::unique_ptr<l1t::PFClusterCollection> l1tpf_calo::SimpleCaloLinkerBase::fetch(
    const edm::OrphanHandle<l1t::PFClusterCollection> &ecal,
    const edm::OrphanHandle<l1t::PFClusterCollection> &hcal) const {
  bool setRefs = (ecal.isValid() && hcal.isValid());
  auto ret = std::make_unique<l1t::PFClusterCollection>();
  l1tpf_calo::GridSelector selector = l1tpf_calo::GridSelector(etaBounds_, phiBounds_, maxClustersEtaPhi_);
  unsigned int index = 0;
  for (const CombinedCluster &cluster : clusters_) {
    index++;
    if (cluster.et > 0) {
      bool photon = (cluster.hcal_et < hoeCut_ * cluster.ecal_et);
      if (photon && noEmInHGC_) {
        if (std::abs(cluster.eta) > 1.5 && std::abs(cluster.eta) < 3.0) {  // 1.5-3 = eta range of HGCal
          continue;
        }
      }
      selector.fill(cluster.et, cluster.eta, cluster.phi, index - 1);
    }
  }
  std::vector<unsigned int> indices = selector.returnSorted();
  for (unsigned int ii = 0; ii < indices.size(); ii++) {
    unsigned int theIndex = indices[ii];
    const CombinedCluster &cluster = clusters_[theIndex];
    bool photon = (cluster.hcal_et < hoeCut_ * cluster.ecal_et);
    if (cluster.et > (photon ? minPhotonEt_ : minHadronEt_)) {
      ret->emplace_back(cluster.et,
                        photon ? cluster.ecal_eta : cluster.eta,
                        photon ? cluster.ecal_phi : cluster.phi,
                        cluster.ecal_et > 0 ? std::max(cluster.et - cluster.ecal_et, 0.f) / cluster.ecal_et : -1,
                        photon);
      if (setRefs) {
        for (const auto &pair : cluster.constituents) {
          assert(pair.first != 0);
          if (pair.first > 0) {  // 1+hcal index
            ret->back().addConstituent(edm::Ptr<l1t::PFCluster>(hcal, +pair.first - 1), pair.second);
          } else {  // -1-ecal index
            ret->back().addConstituent(edm::Ptr<l1t::PFCluster>(ecal, -pair.first + 1), pair.second);
          }
        }
      }
    }
  }
  return ret;
}

l1tpf_calo::SimpleCaloLinker::SimpleCaloLinker(const edm::ParameterSet &pset,
                                               const SingleCaloClusterer &ecal,
                                               const SingleCaloClusterer &hcal)
    : SimpleCaloLinkerBase(pset, ecal, hcal), ecalToHCal_(*grid_) {}

l1tpf_calo::SimpleCaloLinker::~SimpleCaloLinker() {}

void l1tpf_calo::SimpleCaloLinker::clear() {
  clearBase();
  ecalToHCal_.clear();
}

void l1tpf_calo::SimpleCaloLinker::run() {
  unsigned int i, ncells = grid_->size();

  const EtGrid &hraw = hcal_.raw();
  const IndexGrid &ecals = ecal_.indexGrid();
  const IndexGrid &hcals = hcal_.indexGrid();

  // for each ECal cluster, get the corresponding HCal cluster and the sum of the neighbour HCal clusters
  ecalToHCal_.clear();
  for (i = 0; i < ncells; ++i) {
    if (ecals[i] >= 0) {
      if (hcals[i] >= 0) {
        ecalToHCal_[i].ptLocalMax = hcal_.cluster(i).et;
      } else {
        float tot = 0;
        for (int ineigh = 0; ineigh < 8; ++ineigh) {
          tot += hcal_.cluster(grid_->neighbour(i, ineigh)).et;
        }
        ecalToHCal_[i].ptOverNeighLocalMaxSum = tot ? ecal_.cluster(i).et / tot : 0;
      }
    }
  }

  clusterIndex_.fill(-1);
  clusters_.clear();
  CombinedCluster cluster;
  // promote HCal clusters to final clusters
  for (i = 0; i < ncells; ++i) {
    if (hcals[i] >= 0) {
      const Cluster &hcal = hcal_.cluster(i);
      cluster.clear();
      cluster.constituents.emplace_back(+i + 1, 1);
      if (ecalToHCal_[i].ptLocalMax > 0) {
        // direct linking is easy
        const Cluster &ecal = ecal_.cluster(i);
        if (ecal.et + hcal.et > minHadronRawEt_) {
          cluster.ecal_et = ecal.et;
          cluster.hcal_et = hcal.et;
          cluster.et = cluster.ecal_et + cluster.hcal_et;
          float wecal = cluster.ecal_et / cluster.et, whcal = 1.0 - wecal;
          cluster.eta = ecal.eta * wecal + hcal.eta * whcal;
          cluster.phi = ecal.phi * wecal + hcal.phi * whcal;
          cluster.ecal_eta = cluster.eta;
          cluster.ecal_phi = cluster.phi;
          // wrap around phi
          cluster.phi = reco::reduceRange(cluster.phi);
          cluster.constituents.emplace_back(-i - 1, 1);
        }
      } else {
        // sidewas linking is more annonying
        float myet = hcal.et;
        float etot = 0;
        float avg_eta = 0;
        float avg_phi = 0;
        for (int ineigh = 0; ineigh < 8; ++ineigh) {
          int ineighcell = grid_->neighbour(i, ineigh);
          if (ineighcell == -1)
            continue;  // skip dummy cells
          float fracet = myet * ecalToHCal_.neigh(i, ineigh).ptOverNeighLocalMaxSum;
          if (fracet == 0)
            continue;
          etot += fracet;
          avg_eta += fracet * (grid_->eta(ineighcell) - grid_->eta(i));
          avg_phi += fracet * deltaPhi(grid_->phi(ineighcell), grid_->phi(i));
          cluster.constituents.emplace_back(-i - 1, fracet / ecal_.cluster(ineighcell).et);
        }
        if (myet + etot > minHadronRawEt_) {
          cluster.hcal_et = hcal.et;
          cluster.ecal_et = etot;
          cluster.et = myet + etot;
          cluster.eta = hcal.eta + avg_eta / cluster.et;
          cluster.phi = hcal.phi + avg_phi / cluster.et;
          cluster.ecal_eta = cluster.eta;
          cluster.ecal_phi = cluster.phi;
          // wrap around phi
          cluster.phi = reco::reduceRange(cluster.phi);
        }
      }
      if (cluster.et > 0) {
        clusterIndex_[i] = clusters_.size();
        clusters_.push_back(cluster);
      }
    }
  }

  // promote Unlinked ECal clusters to final clusters
  for (i = 0; i < ncells; ++i) {
    if (ecals[i] >= 0 && ecalToHCal_[i].ptLocalMax == 0 && ecalToHCal_[i].ptOverNeighLocalMaxSum == 0) {
      cluster.clear();
      const Cluster &ecal = ecal_.cluster(i);
      cluster.ecal_et = ecal.et;
      cluster.hcal_et = hraw[i];
      cluster.et = cluster.ecal_et + cluster.hcal_et;
      cluster.eta = ecal.eta;
      cluster.phi = ecal.phi;
      cluster.constituents.emplace_back(-i - 1, 1);
      clusterIndex_[i] = clusters_.size();
      clusters_.push_back(cluster);
    }
  }
}

l1tpf_calo::FlatCaloLinker::FlatCaloLinker(const edm::ParameterSet &pset,
                                           const SingleCaloClusterer &ecal,
                                           const SingleCaloClusterer &hcal)
    : SimpleCaloLinkerBase(pset, ecal, hcal), combClusterer_(pset) {}

l1tpf_calo::FlatCaloLinker::~FlatCaloLinker() {}

void l1tpf_calo::FlatCaloLinker::clear() {
  clearBase();
  combClusterer_.clear();
}

void l1tpf_calo::FlatCaloLinker::run() {
  combClusterer_.clear();

  const EtGrid &hraw = hcal_.raw();
  const EtGrid &eraw = ecal_.raw();
  combClusterer_.raw() = eraw;
  combClusterer_.raw() += hraw;

  combClusterer_.run();
  clusterIndex_ = combClusterer_.indexGrid();
  const std::vector<Cluster> &clustersSrc = combClusterer_.clusters();
  unsigned int nclust = clustersSrc.size();
  clusters_.resize(nclust);
  for (unsigned int ic = 0; ic < nclust; ++ic) {
    const Cluster &src = clustersSrc[ic];
    CombinedCluster &dst = clusters_[ic];
    dst.et = src.et;
    dst.eta = src.eta;
    dst.phi = src.phi;
    dst.ecal_eta = src.eta;
    dst.ecal_phi = src.phi;
    dst.ecal_et = 0;
    dst.hcal_et = 0;
    for (const auto &pair : src.constituents) {
      if (eraw[pair.first]) {
        dst.ecal_et += pair.second * eraw[pair.first];
        dst.constituents.emplace_back(-pair.first - 1, pair.second);
      }
      if (hraw[pair.first]) {
        dst.hcal_et += pair.second * hraw[pair.first];
        dst.constituents.emplace_back(+pair.first + 1, pair.second);
      }
    }
  }
}

l1tpf_calo::CombinedCaloLinker::CombinedCaloLinker(const edm::ParameterSet &pset,
                                                   const SingleCaloClusterer &ecal,
                                                   const SingleCaloClusterer &hcal)
    : SimpleCaloLinkerBase(pset, ecal, hcal), combClusterer_(pset) {}

l1tpf_calo::CombinedCaloLinker::~CombinedCaloLinker() {}

void l1tpf_calo::CombinedCaloLinker::clear() {
  clearBase();
  combClusterer_.clear();
}

void l1tpf_calo::CombinedCaloLinker::run() {
  combClusterer_.clear();

  const EtGrid &hraw = hcal_.raw();
  const EtGrid &eraw = ecal_.raw();
  const EtaPhiCenterGrid &eeta = ecal_.etaCenter();
  const EtaPhiCenterGrid &ephi = ecal_.phiCenter();
  combClusterer_.raw() = eraw;
  combClusterer_.raw() += hraw;

  combClusterer_.run();
  clusterIndex_ = combClusterer_.indexGrid();
  const std::vector<Cluster> &clustersSrc = combClusterer_.clusters();
  unsigned int nclust = clustersSrc.size();
  clusters_.resize(nclust);
  for (unsigned int ic = 0; ic < nclust; ++ic) {
    const Cluster &src = clustersSrc[ic];
    CombinedCluster &dst = clusters_[ic];
    dst.et = src.et;
    dst.eta = src.eta;
    dst.phi = src.phi;
    dst.ecal_et = 0;
    dst.hcal_et = 0;
    float pt_max = 0.;
    float eta_ecal = 0.;
    float phi_ecal = 0.;
    for (const auto &pair : src.constituents) {
      if (eraw[pair.first]) {
        float ept = pair.second * eraw[pair.first];
        dst.ecal_et += ept;
        dst.constituents.emplace_back(-pair.first - 1, pair.second);
        if (ept > pt_max) {
          eta_ecal = eeta[pair.first];
          phi_ecal = ephi[pair.first];
          pt_max = ept;
        }
      }
      if (hraw[pair.first]) {
        dst.hcal_et += pair.second * hraw[pair.first];
        dst.constituents.emplace_back(+pair.first + 1, pair.second);
      }
    }
    dst.ecal_eta = eta_ecal;
    dst.ecal_phi = phi_ecal;
  }
}

l1tpf_calo::GridSelector::GridSelector(std::vector<double> etaBounds,
                                       std::vector<double> phiBounds,
                                       std::vector<unsigned int> maxClusters)
    : etaBounds_(etaBounds),
      phiBounds_(phiBounds),
      maxClustersEtaPhi_(maxClusters),
      regionPtIndices_(!maxClusters.empty() ? maxClusters.size() : 1) {}

void l1tpf_calo::GridSelector::fill(float pt, float eta, float phi, unsigned int index) {
  if (!maxClustersEtaPhi_.empty()) {
    unsigned int etai = etaBounds_.size();
    for (unsigned int ie = 0; ie < etaBounds_.size() - 1; ie++) {
      if (eta >= etaBounds_[ie] && eta < etaBounds_[ie + 1]) {
        etai = ie;
        break;
      }
    }
    unsigned int phii = phiBounds_.size();
    for (unsigned int ip = 0; ip < phiBounds_.size() - 1; ip++) {
      if (phi >= phiBounds_[ip] && phi < phiBounds_[ip + 1]) {
        phii = ip;
        break;
      }
    }
    if (etai < etaBounds_.size() && phii < phiBounds_.size()) {
      regionPtIndices_[etai * (phiBounds_.size() - 1) + phii].emplace_back(pt, index);
    }
  } else {
    regionPtIndices_[0].emplace_back(pt, index);
  }
}

std::vector<unsigned int> l1tpf_calo::GridSelector::returnSorted() {
  std::vector<unsigned int> indices;
  for (auto &regionPtIndex : regionPtIndices_) {
    std::sort(regionPtIndex.begin(), regionPtIndex.end(), std::greater<std::pair<float, unsigned int>>());
    for (const auto &p : regionPtIndex) {
      indices.push_back(p.second);
    }
  }
  return indices;
}

std::unique_ptr<l1tpf_calo::SimpleCaloLinkerBase> l1tpf_calo::makeCaloLinker(const edm::ParameterSet &pset,
                                                                             const SingleCaloClusterer &ecal,
                                                                             const SingleCaloClusterer &hcal) {
  const std::string &algo = pset.getParameter<std::string>("algo");
  if (algo == "simple") {
    return std::make_unique<l1tpf_calo::SimpleCaloLinker>(pset, ecal, hcal);
  } else if (algo == "flat") {
    return std::make_unique<l1tpf_calo::FlatCaloLinker>(pset, ecal, hcal);
  } else if (algo == "combined") {
    return std::make_unique<l1tpf_calo::CombinedCaloLinker>(pset, ecal, hcal);
  } else {
    throw cms::Exception("Configuration") << "Unsupported linker algo '" << algo << "'\n";
  }
}
