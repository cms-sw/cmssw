#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "TMath.h"
#include "TVector2.h"
#include <cmath>
using namespace std;

namespace reco {
  namespace MustacheKernel {
    bool inMustache(
        const float maxEta, const float maxPhi, const float ClustE, const float ClusEta, const float ClusPhi) {
      //bool inMust=false;
      //float eta0 = maxEta;
      //float phi0 = maxPhi;

      constexpr float p00 = -0.107537;
      constexpr float p01 = 0.590969;
      constexpr float p02 = -0.076494;
      constexpr float p10 = -0.0268843;
      constexpr float p11 = 0.147742;
      constexpr float p12 = -0.0191235;

      constexpr float w00 = -0.00681785;
      constexpr float w01 = -0.00239516;
      constexpr float w10 = 0.000699995;
      constexpr float w11 = -0.00554331;

      const float sineta0 = std::sin(maxEta);
      const float eta0xsineta0 = maxEta * sineta0;

      //2 parabolas (upper and lower)
      //of the form: y = a*x*x + b

      //b comes from a fit to the width
      //and has a slight dependence on E on the upper edge
      // this only works because of fine tuning :-D
      const float sqrt_log10_clustE = std::sqrt(std::log10(ClustE) + 1.1);
      // we need to have this in two steps, so that we don't improperly shift
      // the lower bound!
      float b_upper = w10 * eta0xsineta0 + w11 / sqrt_log10_clustE;
      float b_lower = w00 * eta0xsineta0 + w01 / sqrt_log10_clustE;
      const float midpoint = 0.5 * (b_upper + b_lower);
      b_upper -= midpoint;
      b_lower -= midpoint;

      //the curvature comes from a parabolic
      //fit for many slices in eta given a
      //slice -0.1 < log10(Et) < 0.1
      const float curv_up = eta0xsineta0 * (p00 * eta0xsineta0 + p01) + p02;
      const float curv_low = eta0xsineta0 * (p10 * eta0xsineta0 + p11) + p12;

      //solving for the curviness given the width of this particular point
      const float a_upper = (1 / (4 * curv_up)) - fabs(b_upper);
      const float a_lower = (1 / (4 * curv_low)) - fabs(b_lower);

      const double dphi = TVector2::Phi_mpi_pi(ClusPhi - maxPhi);
      const double dphi2 = dphi * dphi;
      // minimum offset is half a crystal width in either direction
      // because science.
      const float upper_cut = (std::max((1. / (4. * a_upper)), 0.0) * dphi2 + std::max(b_upper, 0.0087f)) + 0.0087;
      const float lower_cut = (std::max((1. / (4. * a_lower)), 0.0) * dphi2 + std::min(b_lower, -0.0087f));

      //if(deta < upper_cut && deta > lower_cut) inMust=true;

      const float deta = (1 - 2 * (maxEta < 0)) * (ClusEta - maxEta);  // sign flip deta
      return (deta < upper_cut && deta > lower_cut);
    }

    bool inDynamicDPhiWindow(
        const float seedEta, const float seedPhi, const float ClustE, const float ClusEta, const float ClusPhi) {
      // Parameters from the analysis by L. Zygala [https://indico.cern.ch/event/949294/contributions/3988389/attachments/2091573/3514649/2020_08_26_Clustering.pdf]
      constexpr double yoffsetEB = 0.0280506;
      constexpr double scaleEB = 0.946048;
      constexpr double xoffsetEB = -0.101172;
      constexpr double widthEB = 0.432767;

      constexpr double yoffsetEE_0 = 0.0497038;
      constexpr double scaleEE_0 = 0.975707;
      constexpr double xoffsetEE_0 = -0.18149;
      constexpr double widthEE_0 = 0.431729;

      constexpr double yoffsetEE_1 = 0.05643;
      constexpr double scaleEE_1 = 1.60429;
      constexpr double xoffsetEE_1 = -0.642352;
      constexpr double widthEE_1 = 0.458106;

      constexpr double yoffsetEE_2 = 0.0928887;
      constexpr double scaleEE_2 = 1.22321;
      constexpr double xoffsetEE_2 = -0.260256;
      constexpr double widthEE_2 = 0.345852;

      const double absSeedEta = std::abs(seedEta);
      const int etaBin = ((int)(absSeedEta >= 1.479) + (int)(absSeedEta >= 1.75) + (int)(absSeedEta >= 2.0));
      const double logClustEt = std::log10(ClustE / std::cosh(ClusEta));
      const double clusDphi = std::abs(TVector2::Phi_mpi_pi(seedPhi - ClusPhi));

      double yoffset, scale, xoffset, width, saturation, cutoff, maxdphi;

      switch (etaBin) {
        case 0:  // EB
          yoffset = yoffsetEB;
          scale = scaleEB;
          xoffset = xoffsetEB;
          width = 1.0 / widthEB;
          saturation = 0.14;
          cutoff = 0.60;
          break;
        case 1:  // 1.479 -> 1.75
          yoffset = yoffsetEE_0;
          scale = scaleEE_0;
          xoffset = xoffsetEE_0;
          width = 1.0 / widthEE_0;
          saturation = 0.14;
          cutoff = 0.55;
          break;
        case 2:  // 1.75 -> 2.0
          yoffset = yoffsetEE_1;
          scale = scaleEE_1;
          xoffset = xoffsetEE_1;
          width = 1.0 / widthEE_1;
          saturation = 0.12;
          cutoff = 0.45;
          break;
        case 3:  // 2.0 and up
          yoffset = yoffsetEE_2;
          scale = scaleEE_2;
          xoffset = xoffsetEE_2;
          width = 1.0 / widthEE_2;
          saturation = 0.12;
          cutoff = 0.30;
          break;
        default:
          throw cms::Exception("InValidEtaBin")
              << "Calculated invalid eta bin = " << etaBin << " in \"inDynamicDPhiWindow\"" << std::endl;
      }

      maxdphi = yoffset + scale / (1 + std::exp((logClustEt - xoffset) * width));
      maxdphi = std::min(maxdphi, cutoff);
      maxdphi = std::max(maxdphi, saturation);

      return clusDphi < maxdphi;
    }
  }  // namespace MustacheKernel

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
      inMust = MustacheKernel::inMustache(eta0, phi0, (*icl)->energy(), (*icl)->eta(), (*icl)->phi());

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
      bool inMust =
          MustacheKernel::inMustache(eta0, phi0, (clusters[k]).energy(), (clusters[k]).eta(), (clusters[k]).phi());
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
