#include <alpaka/alpaka.hpp>

#include "DataFormats/EgammaReco/interface/alpaka/ElectronSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperClusterDeviceCollection.h"

#include "MagneticField/Portable/interface/ParabolicMagneticField.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ftsFromVertexToPointPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/helixBarrelPlaneCrossingByCircle.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/helixArbitraryPlaneCrossing.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/helixForwardPlaneCrossing.h"

#include "PixelMatchingAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EleRelPointPairPortable.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/Plane.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE T
  getZVtxFromExtrapolation(TAcc const& acc, const Vec3d& primeVtxPos, const Vec3d& hitPos, const Vec3d& candPos) {
    auto sq = [](T x) { return x * x; };

    auto calRDiff2 = [sq](const Vec3d& p1, const Vec3d& p2) { return sq(p2[0] - p1[0]) + sq(p2[1] - p1[1]); };
    const T r1Diff = alpaka::math::sqrt(acc, calRDiff2(primeVtxPos, hitPos));
    const T r2Diff = alpaka::math::sqrt(acc, calRDiff2(hitPos, candPos));

    T zvtx = hitPos[2] - r1Diff * (candPos[2] - hitPos[2]) / r2Diff;
    return zvtx;
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE T
  getCutValue(TAcc const& acc, const T et, const T highEt, const T highEtThres, const T lowEtGrad) {
    return highEt + alpaka::math::min(acc, static_cast<T>(0.), et - highEtThres) * lowEtGrad;
  }

  //--- Kernel for printing the SC SoA
  class PrintSCSoA {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::concepts::Acc<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::SuperClusterDeviceCollection::ConstView view,
                                  int32_t size) const {
      for (int32_t i : uniform_elements(acc, size)) {
        auto sc = view[i];
        printf("For SC i=%d Energy is :%f , theta is :%f , and r is %lf and phi is %lf,  \n",
               i,
               sc.scEnergy(),
               sc.scSeedTheta(),
               sc.scR(),
               sc.scPhi());
        float x = sc.scR() * alpaka::math::sin(acc, sc.scSeedTheta()) * alpaka::math::cos(acc, sc.scPhi());
        float y = sc.scR() * alpaka::math::sin(acc, sc.scSeedTheta()) * alpaka::math::sin(acc, sc.scPhi());
        float z = sc.scR() * alpaka::math::cos(acc, sc.scSeedTheta());
        printf("x: %lf,  y: %lf,  z %lf ", x, z, y);
        Vec3d position{x, y, z};
        printf("  Value of perp2 %lf \n", x * x + y * y);
        printf("Calculate the magnetic field with the parabolic approximation at the SC position : %f\n",
               portableParabolicMagneticField::magneticFieldAtPoint(position));
      }
    }
  };

  //--- Kernel for printing the electron seeds SoA
  class PrintElectronSeedSoA {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::concepts::Acc<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::ElectronSeedDeviceCollection::ConstView view,
                                  int32_t size) const {
      for (int32_t i : uniform_elements(acc, size)) {
        auto seed = view[i];

        printf("For electron seed %d number of hits is %d \n ", i, seed.nHits());
        printf(
            "Seed id: %d is matched? %d and the matchesSC id %d \n", seed.id(), seed.isMatched(), seed.matchedScID());
        printf("Hit 0 is valid: %d\n", seed.hit0isValid());
        printf("Hit 0 detId is : %d\n", seed.hit0detectorID());
        printf("Hit 1 is valid: %d\n", seed.hit1isValid());
        printf("Hit 1 detId is : %d\n", seed.hit1detectorID());
        printf("Hit 0 position is (%lf , %lf , %lf ) \n", seed.hit0Pos().x(), seed.hit0Pos().y(), seed.hit0Pos().z());
        printf("Hit 0 surface is (%lf , %lf , %lf ) \n", seed.surf0Pos().x(), seed.surf0Pos().y(), seed.surf0Pos().z());
        printf(
            "Hit 0 rotation is (%lf , %lf , %lf ) \n", seed.surf0Rot().x(), seed.surf0Rot().y(), seed.surf0Rot().z());
        printf("Hit 1 position is (%lf , %lf , %lf ) \n", seed.hit1Pos().x(), seed.hit1Pos().y(), seed.hit1Pos().z());
        printf("Hit 1 surface is (%lf , %lf , %lf ) \n", seed.surf1Pos().x(), seed.surf1Pos().y(), seed.surf1Pos().z());
        printf(
            "Hit 1 rotation is (%lf , %lf , %lf ) \n", seed.surf1Rot().x(), seed.surf1Rot().y(), seed.surf1Rot().z());
        if (seed.nHits() > 2) {
          printf("Hit 2 is valid: %d", seed.hit2isValid());
          printf("Hit 2 detId is : %d", seed.hit2detectorID());
          printf(
              "Hit 2 position is (%lf , %lf , %lf ) \n ", seed.hit2Pos().x(), seed.hit2Pos().y(), seed.hit2Pos().z());
          printf(
              "Hit 2 surface is (%lf , %lf , %lf ) \n", seed.surf2Pos().x(), seed.surf2Pos().y(), seed.surf2Pos().z());
          printf(
              "Hit 2 rotation is (%lf , %lf , %lf ) \n", seed.surf2Rot().x(), seed.surf2Rot().y(), seed.surf2Rot().z());
        }
      }
    }
  };

  constexpr bool kUseMidpointBField = false;  // use B@SC hit0 midpoint for backward SC->hit0 propagation
  constexpr bool kUseHitBField =
      true;  // use B@hit0 for backward SC->hit0 propagation - this works without changing the validity range of the parabolic parametrized magnetic field

  class SeedToSuperClusterMatcher {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::concepts::Acc<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  reco::ElectronSeedDeviceCollection::View viewEleSeeds,
                                  const int32_t sizeEleSeeds,
                                  reco::SuperClusterDeviceCollection::View viewSCs,
                                  const int32_t sizeSCs,
                                  const double vtx_x,
                                  const double vtx_y,
                                  const double vtx_z) const {
      const Vec3d vertex(vtx_x, vtx_y, vtx_z);

      for (int i : uniform_elements(acc, viewEleSeeds.metadata().size())) {
        auto eleSeed = viewEleSeeds[i];

        if (!(eleSeed.hit0isValid()))
          continue;

        // Access first hit information
        Vec3d hitPosition(eleSeed.hit0Pos().x(), eleSeed.hit0Pos().y(), eleSeed.hit0Pos().z());
        Vec3d surfPosition(eleSeed.surf0Pos().x(), eleSeed.surf0Pos().y(), eleSeed.surf0Pos().z());
        Vec3d surfRotation(eleSeed.surf0Rot().x(), eleSeed.surf0Rot().y(), eleSeed.surf0Rot().z());

        Vec3d hit2Position(eleSeed.hit1Pos().x(), eleSeed.hit1Pos().y(), eleSeed.hit1Pos().z());
        Vec3d surf2Position(eleSeed.surf1Pos().x(), eleSeed.surf1Pos().y(), eleSeed.surf1Pos().z());
        Vec3d surf2Rotation(eleSeed.surf1Rot().x(), eleSeed.surf1Rot().y(), eleSeed.surf1Rot().z());

        for (int j = 0; j < sizeSCs; ++j) {
          const double x = viewSCs[j].scR() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta()) *
                           alpaka::math::cos(acc, viewSCs[j].scPhi());
          const double y = viewSCs[j].scR() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta()) *
                           alpaka::math::sin(acc, viewSCs[j].scPhi());
          const double z = viewSCs[j].scR() * alpaka::math::cos(acc, viewSCs[j].scSeedTheta());

          const float et = viewSCs[j].scEnergy() * alpaka::math::sin(acc, viewSCs[j].scSeedTheta());
          const float e = viewSCs[j].scEnergy();

          Vec3d positionSC(x, y, z);

          for (int charge : {1, -1}) {
            const float c = (charge == 1 ? -2.99792458e-3f : +2.99792458e-3f);

            const float bFieldFirst = kUseHitBField ? portableParabolicMagneticField::magneticFieldAtPoint(hitPosition)
                                      : kUseMidpointBField
                                          ? portableParabolicMagneticField::magneticFieldAtPoint(
                                                Vec3d(0.5 * (positionSC[0] + surfPosition[0]),
                                                      0.5 * (positionSC[1] + surfPosition[1]),
                                                      0.5 * (positionSC[2] + surfPosition[2])))
                                          : portableParabolicMagneticField::magneticFieldAtPoint(positionSC);

            auto newfreeTS = egamma::ftsFromVertexToPoint(acc, positionSC, vertex, e, charge, bFieldFirst);

            const Vec3d position(newfreeTS.get_position());
            const Vec3d momentum(newfreeTS.get_momentum());

            double s = 0;
            bool theSolExists = false;

            Vec3d propagatedPos(0);
            Vec3d propagatedMom(0);

            double rho = (c * bFieldFirst) / momentum.partial_norm(acc);

            // Select propagator by detector ID: BPix -> barrel crossing, FPIX -> forward crossing
            egamma::Plane<typename Vec3d::value_type> plane(surfPosition, surfRotation);
            if (eleSeed.hit0detectorID() == 1) {
              propagators::helixBarrelPlaneCrossing<TAcc, propagators::PropagationDirection::oppositeToMomentum>(
                  acc,
                  position,
                  momentum,
                  rho,
                  surfPosition,
                  surfRotation,
                  theSolExists,
                  propagatedPos,
                  propagatedMom,
                  s);
            } else {
              propagators::helixForwardPlaneCrossing<TAcc, propagators::PropagationDirection::oppositeToMomentum>(
                  acc, position, momentum, rho, plane, s, propagatedPos, propagatedMom, theSolExists);
            }

            if (!theSolExists)
              continue;

            propagatedMom *= momentum.norm(acc) / propagatedMom.norm(acc);

            egamma::EleRelPointPairPortable<typename Vec3d::value_type> pair(hitPosition, propagatedPos, vertex);

            const float dPhiMax = getCutValue(acc, et, 0.05f, 20.f, -0.002f);
            const float dRZMax = getCutValue(acc, et, 9999.f, 0.f, 0.f);
            const float dRZ = eleSeed.hit0detectorID() != 1 ? pair.dPerp(acc) : pair.dZ();
            const float dPhi = pair.dPhi(acc);

            if ((dPhiMax >= 0 && alpaka::math::abs(acc, dPhi) > dPhiMax) ||
                (dRZMax >= 0 && alpaka::math::abs(acc, dRZ) > dRZMax))
              continue;

            const double zVertex =
                getZVtxFromExtrapolation<TAcc, typename Vec3d::value_type>(acc, vertex, hitPosition, positionSC);
            Vec3d vertexUpdated(vertex[0], vertex[1], zVertex);

            // --- Second hit ---
            if (!(eleSeed.hit1isValid()))
              continue;

            const float bFieldHit0 = portableParabolicMagneticField::magneticFieldAtPoint(hitPosition);
            auto firstMatchFreeTraj =
                egamma::ftsFromVertexToPoint(acc, hitPosition, vertexUpdated, e, charge, bFieldHit0);
            Vec3d position2(firstMatchFreeTraj.get_position());
            Vec3d momentum2(firstMatchFreeTraj.get_momentum());

            rho = (c * bFieldHit0) / momentum2.partial_norm(acc);

            theSolExists = false;
            propagatedPos = Vec3d(0);
            propagatedMom = Vec3d(0);

            egamma::Plane<typename Vec3d::value_type> plane2(surf2Position, surf2Rotation);
            if (eleSeed.hit1detectorID() == 1) {
              propagators::helixBarrelPlaneCrossing<TAcc, propagators::PropagationDirection::alongMomentum>(
                  acc,
                  position2,
                  momentum2,
                  rho,
                  surf2Position,
                  surf2Rotation,
                  theSolExists,
                  propagatedPos,
                  propagatedMom,
                  s);
            } else {
              propagators::helixForwardPlaneCrossing<TAcc, propagators::PropagationDirection::alongMomentum>(
                  acc, position2, momentum2, rho, plane2, s, propagatedPos, propagatedMom, theSolExists);
            }

            if (!theSolExists)
              continue;

            propagatedMom *= momentum2.norm(acc) / propagatedMom.norm(acc);

            egamma::EleRelPointPairPortable<typename Vec3d::value_type> pair2(
                hit2Position, propagatedPos, vertexUpdated);

            const float dPhiMax2 = getCutValue(acc, et, 0.003f, 0.f, 0.f);
            const float dRZMax2 = getCutValue(acc, et, 0.05f, 30.f, -0.002f);
            const float dRZ2 = eleSeed.hit1detectorID() != 1 ? pair2.dPerp(acc) : pair2.dZ();
            const float dPhi2 = pair2.dPhi(acc);

            if ((dPhiMax2 >= 0 && alpaka::math::abs(acc, dPhi2) > dPhiMax2) ||
                (dRZMax2 >= 0 && alpaka::math::abs(acc, dRZ2) > dRZMax2))
              continue;

            float dRZ3 = 0;
            float dPhi3 = 0;
            // --- Third hit (triplet seeds only) ---
            if (eleSeed.nHits() > 2 && eleSeed.hit2isValid()) {
              Vec3d hit3Position(eleSeed.hit2Pos().x(), eleSeed.hit2Pos().y(), eleSeed.hit2Pos().z());
              Vec3d surf3Position(eleSeed.surf2Pos().x(), eleSeed.surf2Pos().y(), eleSeed.surf2Pos().z());
              Vec3d surf3Rotation(eleSeed.surf2Rot().x(), eleSeed.surf2Rot().y(), eleSeed.surf2Rot().z());

              bool thirdSolExists = false;
              Vec3d propagatedPos3(0), propagatedMom3(0);
              double s3 = 0;

              egamma::Plane<typename Vec3d::value_type> plane3(surf3Position, surf3Rotation);
              if (eleSeed.hit2detectorID() == 1) {
                propagators::helixBarrelPlaneCrossing<TAcc, propagators::PropagationDirection::alongMomentum>(
                    acc,
                    position2,
                    momentum2,
                    rho,
                    surf3Position,
                    surf3Rotation,
                    thirdSolExists,
                    propagatedPos3,
                    propagatedMom3,
                    s3);
              } else {
                propagators::helixForwardPlaneCrossing<TAcc, propagators::PropagationDirection::alongMomentum>(
                    acc, position2, momentum2, rho, plane3, s3, propagatedPos3, propagatedMom3, thirdSolExists);
              }

              if (!thirdSolExists)
                continue;

              egamma::EleRelPointPairPortable<typename Vec3d::value_type> pair3(
                  hit3Position, propagatedPos3, vertexUpdated);

              const float dPhiMax3 = getCutValue(acc, et, 0.003f, 0.f, 0.f);
              const float dRZMax3 = getCutValue(acc, et, 0.05f, 30.f, -0.002f);
              dRZ3 = eleSeed.hit2detectorID() != 1 ? pair3.dPerp(acc) : pair3.dZ();
              dPhi3 = pair3.dPhi(acc);

              if ((dPhiMax3 >= 0 && alpaka::math::abs(acc, dPhi3) > dPhiMax3) ||
                  (dRZMax3 >= 0 && alpaka::math::abs(acc, dRZ3) > dRZMax3))
                continue;
            }

            eleSeed.matchedScID() = static_cast<int16_t>(viewSCs[j].id());
            eleSeed.isMatched() = static_cast<int16_t>(1);

            using EVector3f = Eigen::Matrix<float, 3, 1>;
            if (charge == 1) {
              eleSeed.PMVars_dRZPos() = EVector3f(dRZ, dRZ2, dRZ3);
              eleSeed.PMVars_dPhiPos() = EVector3f(dPhi, dPhi2, dPhi3);
            } else {
              eleSeed.PMVars_dRZNeg() = EVector3f(dRZ, dRZ2, dRZ3);
              eleSeed.PMVars_dPhiNeg() = EVector3f(dPhi, dPhi2, dPhi3);
            }
          }
        }
      }
    }
  };

  //---- Kernel launch for printing the SC SoA collection
  void PixelMatchingAlgo::printEleSeeds(Queue& queue, const reco::ElectronSeedDeviceCollection& collection) const {
    uint32_t items = 32;
    uint32_t groups = divide_up_by(collection->metadata().size(), items);
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, PrintElectronSeedSoA{}, collection.view(), collection->metadata().size());
  }

  //---- Kernel launch for printing the SC SoA collection
  void PixelMatchingAlgo::printSCs(Queue& queue, const reco::SuperClusterDeviceCollection& collection) const {
    uint32_t items = 32;
    uint32_t groups = divide_up_by(collection->metadata().size(), items);
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, PrintSCSoA{}, collection.view(), collection->metadata().size());
  }

  //---- Kernel launch for SC and seed matching
  void PixelMatchingAlgo::matchSeeds(Queue& queue,
                                     reco::ElectronSeedDeviceCollection& collection,
                                     reco::SuperClusterDeviceCollection& collectionSCs,
                                     double vtx_X,
                                     double vtx_Y,
                                     double vtx_Z) const {
    uint32_t items = 32;
    auto nSeeds = static_cast<uint32_t>(collection->metadata().size());
    uint32_t groups = divide_up_by(nSeeds, items);

    if (groups < 1)
      return;
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SeedToSuperClusterMatcher{},
                        collection.view(),
                        collection->metadata().size(),
                        collectionSCs.view(),
                        collectionSCs->metadata().size(),
                        vtx_X,
                        vtx_Y,
                        vtx_Z);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
