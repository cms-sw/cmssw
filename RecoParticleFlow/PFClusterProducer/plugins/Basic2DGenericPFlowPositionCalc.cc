#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include "vdt/vdtMath.h"

#include <boost/iterator/function_output_iterator.hpp>

#include <cmath>
#include <iterator>
#include <memory>
#include <tuple>

class Basic2DGenericPFlowPositionCalc : public PFCPositionCalculatorBase {
public:
  Basic2DGenericPFlowPositionCalc(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : PFCPositionCalculatorBase(conf, cc),
        _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
        _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")) {
    std::vector<int> detectorEnum;
    std::vector<int> depths;
    std::vector<double> logWeightDenom;
    std::vector<float> logWeightDenomInv;

    if (conf.exists("logWeightDenominatorByDetector")) {
      const std::vector<edm::ParameterSet>& logWeightDenominatorByDetectorPSet =
          conf.getParameterSetVector("logWeightDenominatorByDetector");

      for (const auto& pset : logWeightDenominatorByDetectorPSet) {
        if (!pset.exists("detector")) {
          throw cms::Exception("logWeightDenominatorByDetectorPSet") << "logWeightDenominator : detector not specified";
        }

        const std::string& det = pset.getParameter<std::string>("detector");

        if (det == std::string("HCAL_BARREL1") || det == std::string("HCAL_ENDCAP")) {
          std::vector<int> depthsT = pset.getParameter<std::vector<int> >("depths");
          std::vector<double> logWeightDenomT = pset.getParameter<std::vector<double> >("logWeightDenominator");
          if (logWeightDenomT.size() != depthsT.size()) {
            throw cms::Exception("logWeightDenominator") << "logWeightDenominator mismatch with the numbers of depths";
          }
          for (unsigned int i = 0; i < depthsT.size(); ++i) {
            if (det == std::string("HCAL_BARREL1"))
              detectorEnum.push_back(1);
            if (det == std::string("HCAL_ENDCAP"))
              detectorEnum.push_back(2);
            depths.push_back(depthsT[i]);
            logWeightDenom.push_back(logWeightDenomT[i]);
          }
        }
      }
    } else {
      detectorEnum.push_back(0);
      depths.push_back(0);
      logWeightDenom.push_back(conf.getParameter<double>("logWeightDenominator"));
    }

    for (unsigned int i = 0; i < depths.size(); ++i) {
      logWeightDenomInv.push_back(1. / logWeightDenom[i]);
    }

    //    _logWeightDenom = std::make_pair(depths,logWeightDenomInv);
    _logWeightDenom = std::make_tuple(detectorEnum, depths, logWeightDenomInv);

    _timeResolutionCalcBarrel.reset(nullptr);
    if (conf.exists("timeResolutionCalcBarrel")) {
      const edm::ParameterSet& timeResConf = conf.getParameterSet("timeResolutionCalcBarrel");
      _timeResolutionCalcBarrel = std::make_unique<CaloRecHitResolutionProvider>(timeResConf);
    }
    _timeResolutionCalcEndcap.reset(nullptr);
    if (conf.exists("timeResolutionCalcEndcap")) {
      const edm::ParameterSet& timeResConf = conf.getParameterSet("timeResolutionCalcEndcap");
      _timeResolutionCalcEndcap = std::make_unique<CaloRecHitResolutionProvider>(timeResConf);
    }

    switch (_posCalcNCrystals) {
      case 5:
      case 9:
      case -1:
        break;
      default:
        edm::LogError("Basic2DGenericPFlowPositionCalc") << "posCalcNCrystals not valid";
        assert(0);  // bug
    }
  }

  Basic2DGenericPFlowPositionCalc(const Basic2DGenericPFlowPositionCalc&) = delete;
  Basic2DGenericPFlowPositionCalc& operator=(const Basic2DGenericPFlowPositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&) override;
  void calculateAndSetPositions(reco::PFClusterCollection&) override;

private:
  const int _posCalcNCrystals;
  std::tuple<std::vector<int>, std::vector<int>, std::vector<float> > _logWeightDenom;
  const float _minAllowedNorm;

  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcBarrel;
  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcEndcap;

  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory, Basic2DGenericPFlowPositionCalc, "Basic2DGenericPFlowPositionCalc");

namespace {
  inline bool isBarrel(int cell_layer) {
    return (cell_layer == PFLayer::HCAL_BARREL1 || cell_layer == PFLayer::HCAL_BARREL2 ||
            cell_layer == PFLayer::ECAL_BARREL);
  }
}  // namespace

void Basic2DGenericPFlowPositionCalc::calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void Basic2DGenericPFlowPositionCalc::calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for (reco::PFCluster& cluster : clusters) {
    calculateAndSetPositionActual(cluster);
  }
}

void Basic2DGenericPFlowPositionCalc::calculateAndSetPositionActual(reco::PFCluster& cluster) const {
  if (!cluster.seed()) {
    throw cms::Exception("ClusterWithNoSeed") << " Found a cluster with no seed: " << cluster;
  }
  double cl_energy = 0;
  double cl_time = 0;
  double cl_timeweight = 0.0;
  double max_e = 0.0;
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  // find the seed and max layer and also calculate time
  //Michalis : Even if we dont use timing in clustering here we should fill
  //the time information for the cluster. This should use the timing resolution(1/E)
  //so the weight should be fraction*E^2
  //calculate a simplistic depth now. The log weighted will be done
  //in different stage

  auto const recHitCollection =
      &(*cluster.recHitFractions()[0].recHitRef()) - cluster.recHitFractions()[0].recHitRef().key();
  auto nhits = cluster.recHitFractions().size();
  struct LHit {
    reco::PFRecHit const* hit;
    float energy;
    float fraction;
  };
  declareDynArray(LHit, nhits, hits);
  for (auto i = 0U; i < nhits; ++i) {
    auto const& hf = cluster.recHitFractions()[i];
    auto k = hf.recHitRef().key();
    auto p = recHitCollection + k;
    hits[i] = {p, (*p).energy(), float(hf.fraction())};
  }

  bool resGiven = bool(_timeResolutionCalcBarrel) && bool(_timeResolutionCalcEndcap);
  LHit mySeed = {};
  for (auto const& rhf : hits) {
    const reco::PFRecHit& refhit = *rhf.hit;
    if (refhit.detId() == cluster.seed())
      mySeed = rhf;
    const auto rh_fraction = rhf.fraction;
    const auto rh_rawenergy = rhf.energy;
    const auto rh_energy = rh_rawenergy * rh_fraction;
#ifdef PF_DEBUG
    if UNLIKELY (edm::isNotFinite(rh_energy)) {
      throw cms::Exception("PFClusterAlgo") << "rechit " << refhit.detId() << " has a NaN energy... "
                                            << "The input of the particle flow clustering seems to be corrupted.";
    }
#endif
    cl_energy += rh_energy;
    // If time resolution is given, calculated weighted average
    if (resGiven) {
      double res2 = 1.e-4;
      int cell_layer = (int)refhit.layer();
      res2 = isBarrel(cell_layer) ? 1. / _timeResolutionCalcBarrel->timeResolution2(rh_rawenergy)
                                  : 1. / _timeResolutionCalcEndcap->timeResolution2(rh_rawenergy);
      cl_time += rh_fraction * refhit.time() * res2;
      cl_timeweight += rh_fraction * res2;
    } else {  // assume resolution = 1/E**2
      const double rh_rawenergy2 = rh_rawenergy * rh_rawenergy;
      cl_timeweight += rh_rawenergy2 * rh_fraction;
      cl_time += rh_rawenergy2 * rh_fraction * refhit.time();
    }

    if (rh_energy > max_e) {
      max_e = rh_energy;
      max_e_layer = refhit.layer();
    }
  }

  cluster.setEnergy(cl_energy);
  cluster.setTime(cl_time / cl_timeweight);
  if (resGiven) {
    cluster.setTimeError(std::sqrt(1.0f / float(cl_timeweight)));
  }
  cluster.setLayer(max_e_layer);

  // calculate the position
  bool single_depth = true;
  int ref_depth = -1;
  double depth = 0.0;
  double position_norm = 0.0;
  double x(0.0), y(0.0), z(0.0);
  if (nullptr != mySeed.hit) {
    auto seedNeighbours = mySeed.hit->neighbours();
    switch (_posCalcNCrystals) {
      case 5:
        seedNeighbours = mySeed.hit->neighbours4();
        break;
      case 9:
        seedNeighbours = mySeed.hit->neighbours8();
        break;
      default:
        break;
    }

    auto compute = [&](LHit const& rhf) {
      const reco::PFRecHit& refhit = *rhf.hit;

      int cell_layer = (int)refhit.layer();
      float threshold = 0;

      for (unsigned int j = 0; j < (std::get<2>(_logWeightDenom)).size(); ++j) {
        // barrel is detecor type1
        int detectorEnum = std::get<0>(_logWeightDenom)[j];
        int depth = std::get<1>(_logWeightDenom)[j];

        if ((cell_layer == PFLayer::HCAL_BARREL1 && detectorEnum == 1 && refhit.depth() == depth) ||
            (cell_layer == PFLayer::HCAL_ENDCAP && detectorEnum == 2 && refhit.depth() == depth) || detectorEnum == 0)
          threshold = std::get<2>(_logWeightDenom)[j];
      }

      if (ref_depth < 0)
        ref_depth = refhit.depth();  // Initialize reference depth
      else if (refhit.depth() != ref_depth) {
        // Found a rechit with a different depth
        single_depth = false;
      }
      const auto rh_energy = rhf.energy * rhf.fraction;
      const auto norm =
          (rhf.fraction < _minFractionInCalc ? 0.0f : std::max(0.0f, vdt::fast_logf(rh_energy * threshold)));
      const auto rhpos_xyz = refhit.position() * norm;
      x += rhpos_xyz.x();
      y += rhpos_xyz.y();
      z += rhpos_xyz.z();
      depth += refhit.depth() * norm;
      position_norm += norm;
    };

    if (_posCalcNCrystals != -1)  // sorted to make neighbour search faster (maybe)
      std::sort(hits.begin(), hits.end(), [](LHit const& a, LHit const& b) { return a.hit < b.hit; });

    if (_posCalcNCrystals == -1)
      for (auto const& rhf : hits)
        compute(rhf);
    else {  // only seed and its neighbours
      compute(mySeed);
      // search seedNeighbours to find energy fraction in cluster (sic)
      unInitDynArray(reco::PFRecHit const*, seedNeighbours.size(), nei);
      for (auto k : seedNeighbours) {
        nei.push_back(recHitCollection + k);
      }
      std::sort(nei.begin(), nei.end());
      struct LHitLess {
        auto operator()(LHit const& a, reco::PFRecHit const* b) const { return a.hit < b; }
        auto operator()(reco::PFRecHit const* b, LHit const& a) const { return b < a.hit; }
      };
      std::set_intersection(
          hits.begin(), hits.end(), nei.begin(), nei.end(), boost::make_function_output_iterator(compute), LHitLess());
    }
  } else {
    throw cms::Exception("Basic2DGenerticPFlowPositionCalc")
        << "Cluster seed hit is null, something is wrong with PFlow RecHit!";
  }

  if (position_norm < _minAllowedNorm) {
    edm::LogError("WeirdClusterNormalization") << "PFCluster too far from seeding cell: set position to (0,0,0).";
    cluster.setPosition(math::XYZPoint(0, 0, 0));
    cluster.calculatePositionREP();
  } else {
    const double norm_inverse = 1.0 / position_norm;
    x *= norm_inverse;
    y *= norm_inverse;
    z *= norm_inverse;
    if (single_depth)
      depth = ref_depth;
    else
      depth *= norm_inverse;
    cluster.setPosition(math::XYZPoint(x, y, z));
    cluster.setDepth(depth);
    cluster.calculatePositionREP();
  }
}
