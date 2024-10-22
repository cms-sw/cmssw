#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "TVector2.h"
#include <cmath>
using namespace std;

namespace reco {
  namespace MustacheKernel {
    bool inMustache(const EcalMustacheSCParameters* params,
                    const float maxEta,
                    const float maxPhi,
                    const float ClustE,
                    const float ClusEta,
                    const float ClusPhi) {
      const auto log10ClustE = std::log10(ClustE);
      const auto parabola_params = params->parabolaParameters(log10ClustE, std::abs(ClusEta));
      if (!parabola_params) {
        return false;
      }

      const float sineta0 = std::sin(maxEta);
      const float eta0xsineta0 = maxEta * sineta0;

      //2 parabolas (upper and lower)
      //of the form: y = a*x*x + b

      //b comes from a fit to the width
      //and has a slight dependence on E on the upper edge
      // this only works because of fine tuning :-D
      const float sqrt_log10_clustE = std::sqrt(log10ClustE + params->sqrtLogClustETuning());
      const float b_upper =
          parabola_params->w1Up[0] * eta0xsineta0 + parabola_params->w1Up[1] / sqrt_log10_clustE -
          0.5 * (parabola_params->w1Up[0] * eta0xsineta0 + parabola_params->w1Up[1] / sqrt_log10_clustE +
                 parabola_params->w0Up[0] * eta0xsineta0 + parabola_params->w0Up[1] / sqrt_log10_clustE);
      const float b_lower =
          parabola_params->w0Low[0] * eta0xsineta0 + parabola_params->w0Low[1] / sqrt_log10_clustE -
          0.5 * (parabola_params->w1Low[0] * eta0xsineta0 + parabola_params->w1Low[1] / sqrt_log10_clustE +
                 parabola_params->w0Low[0] * eta0xsineta0 + parabola_params->w0Low[1] / sqrt_log10_clustE);

      //the curvature comes from a parabolic
      //fit for many slices in eta given a
      //slice -0.1 < log10(Et) < 0.1
      const float curv_up =
          eta0xsineta0 * (parabola_params->pUp[0] * eta0xsineta0 + parabola_params->pUp[1]) + parabola_params->pUp[2];
      const float curv_low = eta0xsineta0 * (parabola_params->pLow[0] * eta0xsineta0 + parabola_params->pLow[1]) +
                             parabola_params->pLow[2];

      //solving for the curviness given the width of this particular point
      const float a_upper = (1. / (4. * curv_up)) - std::abs(b_upper);
      const float a_lower = (1. / (4. * curv_low)) - std::abs(b_lower);

      const double dphi = TVector2::Phi_mpi_pi(ClusPhi - maxPhi);
      const double dphi2 = dphi * dphi;
      // minimum offset is half a crystal width in either direction
      // because science.
      constexpr float half_crystal_width = 0.0087;
      const float upper_cut =
          (std::max((1. / (4. * a_upper)), 0.0) * dphi2 + std::max(b_upper, half_crystal_width)) + half_crystal_width;
      const float lower_cut = (std::max((1. / (4. * a_lower)), 0.0) * dphi2 + std::min(b_lower, -half_crystal_width));

      const float deta = (1 - 2 * (maxEta < 0)) * (ClusEta - maxEta);  // sign flip deta
      return (deta < upper_cut && deta > lower_cut);
    }

    bool inDynamicDPhiWindow(const EcalSCDynamicDPhiParameters* params,
                             const float seedEta,
                             const float seedPhi,
                             const float ClustE,
                             const float ClusEta,
                             const float ClusPhi) {
      const double absSeedEta = std::abs(seedEta);
      const double logClustEt = std::log10(ClustE / std::cosh(ClusEta));
      const double clusDphi = std::abs(TVector2::Phi_mpi_pi(seedPhi - ClusPhi));

      const auto dynamicDPhiParams = params->dynamicDPhiParameters(ClustE, absSeedEta);
      if (!dynamicDPhiParams) {
        return false;
      }

      auto maxdphi = dynamicDPhiParams->yoffset +
                     dynamicDPhiParams->scale /
                         (1. + std::exp((logClustEt - dynamicDPhiParams->xoffset) / dynamicDPhiParams->width));
      maxdphi = std::min(maxdphi, dynamicDPhiParams->cutoff);
      maxdphi = std::max(maxdphi, dynamicDPhiParams->saturation);

      return clusDphi < maxdphi;
    }
  }  // namespace MustacheKernel

  Mustache::Mustache(const EcalMustacheSCParameters* mustache_params) : mustache_params_(mustache_params) {}

  void Mustache::MustacheID(const reco::SuperCluster& sc, int& nclusters, float& EoutsideMustache) {
    MustacheID(sc.clustersBegin(), sc.clustersEnd(), nclusters, EoutsideMustache);
  }

  void Mustache::MustacheID(const CaloClusterPtrVector& clusters, int& nclusters, float& EoutsideMustache) {
    MustacheID(clusters.begin(), clusters.end(), nclusters, EoutsideMustache);
  }

  void Mustache::MustacheID(const std::vector<const CaloCluster*>& clusters, int& nclusters, float& EoutsideMustache) {
    MustacheID(clusters.cbegin(), clusters.cend(), nclusters, EoutsideMustache);
  }

  template <class RandomAccessPtrIterator>
  void Mustache::MustacheID(const RandomAccessPtrIterator& begin,
                            const RandomAccessPtrIterator& end,
                            int& nclusters,
                            float& EoutsideMustache) {
    nclusters = 0;
    EoutsideMustache = 0;

    unsigned int ncl = end - begin;
    if (!ncl)
      return;

    //loop over all clusters to find the one with highest energy
    RandomAccessPtrIterator icl = begin;
    RandomAccessPtrIterator clmax = end;
    float emax = 0;
    for (; icl != end; ++icl) {
      const float e = (*icl)->energy();
      if (e > emax) {
        emax = e;
        clmax = icl;
      }
    }

    if (end == clmax)
      return;

    float eta0 = (*clmax)->eta();
    float phi0 = (*clmax)->phi();

    bool inMust = false;
    icl = begin;
    for (; icl != end; ++icl) {
      inMust = MustacheKernel::inMustache(mustache_params_, eta0, phi0, (*icl)->energy(), (*icl)->eta(), (*icl)->phi());

      nclusters += (int)!inMust;
      EoutsideMustache += (!inMust) * ((*icl)->energy());
    }
  }

  void Mustache::MustacheClust(const std::vector<CaloCluster>& clusters,
                               std::vector<unsigned int>& insideMust,
                               std::vector<unsigned int>& outsideMust) {
    unsigned int ncl = clusters.size();
    if (!ncl)
      return;

    //loop over all clusters to find the one with highest energy
    float emax = 0;
    int imax = -1;
    for (unsigned int i = 0; i < ncl; ++i) {
      float e = (clusters[i]).energy();
      if (e > emax) {
        emax = e;
        imax = i;
      }
    }

    if (imax < 0)
      return;
    float eta0 = (clusters[imax]).eta();
    float phi0 = (clusters[imax]).phi();

    for (unsigned int k = 0; k < ncl; k++) {
      bool inMust = MustacheKernel::inMustache(
          mustache_params_, eta0, phi0, (clusters[k]).energy(), (clusters[k]).eta(), (clusters[k]).phi());
      //return indices of Clusters outside the Mustache
      if (!(inMust)) {
        outsideMust.push_back(k);
      } else {  //return indices of Clusters inside the Mustache
        insideMust.push_back(k);
      }
    }
  }

  void Mustache::FillMustacheVar(const std::vector<CaloCluster>& clusters) {
    Energy_In_Mustache_ = 0;
    Energy_Outside_Mustache_ = 0;
    LowestClusterEInMustache_ = 0;
    excluded_ = 0;
    included_ = 0;
    std::multimap<float, unsigned int> OrderedClust;
    std::vector<unsigned int> insideMust;
    std::vector<unsigned int> outsideMust;
    MustacheClust(clusters, insideMust, outsideMust);
    included_ = insideMust.size();
    excluded_ = outsideMust.size();
    for (unsigned int i = 0; i < insideMust.size(); ++i) {
      unsigned int index = insideMust[i];
      Energy_In_Mustache_ = clusters[index].energy() + Energy_In_Mustache_;
      OrderedClust.insert(make_pair(clusters[index].energy(), index));
    }
    for (unsigned int i = 0; i < outsideMust.size(); ++i) {
      unsigned int index = outsideMust[i];
      Energy_Outside_Mustache_ = clusters[index].energy() + Energy_Outside_Mustache_;
      Et_Outside_Mustache_ = clusters[index].energy() * sin(clusters[index].position().theta()) + Et_Outside_Mustache_;
    }
    std::multimap<float, unsigned int>::iterator it;
    it = OrderedClust.begin();
    unsigned int lowEindex = (*it).second;
    LowestClusterEInMustache_ = clusters[lowEindex].energy();
  }
}  // namespace reco
