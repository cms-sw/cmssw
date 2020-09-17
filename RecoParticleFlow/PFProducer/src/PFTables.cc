#include "RecoParticleFlow/PFProducer/interface/PFTables.h"

using namespace edm::soa::col;
namespace edm::soa {

  TrackTableVertex makeTrackTableVertex(std::vector<reco::PFBlockElement*> const& objects) {
    return {objects,
            edm::soa::column_fillers(
                pf::track::ExtrapolationValid::filler([](reco::PFBlockElement* x) {
                  return x->trackRefPF()->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).isValid();
                }),
                pf::track::Pt::filler([](reco::PFBlockElement* x) {
                  return sqrt(x->trackRefPF()
                                  ->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach)
                                  .momentum()
                                  .Vect()
                                  .Perp2());
                }),
                pf::track::IsLinkedToDisplacedVertex::filler(
                    [](reco::PFBlockElement* x) { return x->isLinkedToDisplacedVertex(); }),
                pf::track::TrackType_FROM_GAMMACONV::filler(
                    [](reco::PFBlockElement* x) { return x->trackType(reco::PFBlockElement::T_FROM_GAMMACONV); }),
                pf::track::TrackType_FROM_V0::filler(
                    [](reco::PFBlockElement* x) { return x->trackType(reco::PFBlockElement::T_FROM_V0); }),
                pf::track::V0RefIsNonNull::filler([](reco::PFBlockElement* x) { return x->V0Ref().isNonnull(); }),
                pf::track::V0RefKey::filler([](reco::PFBlockElement* x) { return refToElementID(x->V0Ref()); }),
                pf::track::DisplacedVertexRef_TO_DISP_IsNonNull::filler([](reco::PFBlockElement* x) {
                  return x->displacedVertexRef(reco::PFBlockElement::T_TO_DISP).isNonnull();
                }),
                pf::track::DisplacedVertexRef_TO_DISP_Key::filler([](reco::PFBlockElement* x) {
                  return refToElementID(x->displacedVertexRef(reco::PFBlockElement::T_TO_DISP));
                }),
                pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull::filler([](reco::PFBlockElement* x) {
                  return x->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP).isNonnull();
                }),
                pf::track::DisplacedVertexRef_FROM_DISP_Key::filler([](reco::PFBlockElement* x) {
                  return refToElementID(x->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP));
                }))};
  }

  ConvRefTable makeConvRefTable(const std::vector<reco::ConversionRef>& convrefs) {
    return {convrefs,
            edm::soa::column_fillers(
                pf::track::ConvRefIsNonNull::filler([](const reco::ConversionRef& x) { return x.isNonnull(); }),
                pf::track::ConvRefKey::filler([](const reco::ConversionRef& x) { return refToElementID(x); }))};
  }

  TrackTableExtrapolation makeTrackTable(std::vector<reco::PFBlockElement*> const& targetSet,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    std::vector<bool> valid;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> r;

    for (auto pfelement_track : targetSet) {
      const reco::PFRecTrackRef& trackref = pfelement_track->trackRefPF();

      const reco::PFTrajectoryPoint& point = trackref->extrapolatedPoint(layerType);

      valid.push_back(point.isValid());

      if (point.isValid()) {
        eta.push_back(point.positionREP().eta());
        phi.push_back(point.positionREP().phi());
        x.push_back(point.position().X());
        y.push_back(point.position().Y());
        z.push_back(point.position().Z());
        r.push_back(point.position().R());
      } else {
        eta.push_back(0);
        phi.push_back(0);
        x.push_back(0);
        y.push_back(0);
        z.push_back(0);
        r.push_back(0);
      }
    }

    return TrackTableExtrapolation(valid, eta, phi, x, y, z, r);
  }

  //this should be unified with the function above
  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementGsfTrack*> const& targetSet,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    std::vector<bool> valid;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> r;

    for (auto pfelement_track : targetSet) {
      const auto& trackref = pfelement_track->GsftrackPF();

      const reco::PFTrajectoryPoint& point = trackref.extrapolatedPoint(layerType);

      valid.push_back(point.isValid());

      if (point.isValid()) {
        eta.push_back(point.positionREP().eta());
        phi.push_back(point.positionREP().phi());
        x.push_back(point.position().X());
        y.push_back(point.position().Y());
        z.push_back(point.position().Z());
        r.push_back(point.position().R());
      } else {
        eta.push_back(0);
        phi.push_back(0);
        x.push_back(0);
        y.push_back(0);
        z.push_back(0);
        r.push_back(0);
      }
    }

    return TrackTableExtrapolation(valid, eta, phi, x, y, z, r);
  }

  RecHitTable makeRecHitTable(std::vector<const reco::PFRecHitFraction*> const& objects) {
    return {
        objects,
        edm::soa::column_fillers(
            pf::rechit::DetIdValue::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->detId(); }),
            pf::rechit::Fraction::filler([](reco::PFRecHitFraction const* x) { return x->fraction(); }),
            pf::rechit::Eta::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->positionREP().eta(); }),
            pf::rechit::Phi::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->positionREP().phi(); }),
            pf::rechit::Posx::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().x(); }),
            pf::rechit::Posy::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().y(); }),
            pf::rechit::Posz::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().z(); }),

            pf::rechit::Corner0x::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[0].x(); }),
            pf::rechit::Corner0y::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[0].y(); }),
            pf::rechit::Corner0z::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[0].z(); }),
            pf::rechit::Corner1x::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[1].x(); }),
            pf::rechit::Corner1y::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[1].y(); }),
            pf::rechit::Corner1z::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[1].z(); }),
            pf::rechit::Corner2x::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[2].x(); }),
            pf::rechit::Corner2y::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[2].y(); }),
            pf::rechit::Corner2z::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[2].z(); }),
            pf::rechit::Corner3x::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[3].x(); }),
            pf::rechit::Corner3y::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[3].y(); }),
            pf::rechit::Corner3z::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[3].z(); }),

            pf::rechit::Corner0eta::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[0].eta(); }),
            pf::rechit::Corner0phi::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[0].phi(); }),
            pf::rechit::Corner1eta::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[1].eta(); }),
            pf::rechit::Corner1phi::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[1].phi(); }),
            pf::rechit::Corner2eta::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[2].eta(); }),
            pf::rechit::Corner2phi::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[2].phi(); }),
            pf::rechit::Corner3eta::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[3].eta(); }),
            pf::rechit::Corner3phi::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersREP()[3].phi(); }),

            pf::rechit::Corner0xBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[0].basicVector().x(); }),
            pf::rechit::Corner0yBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[0].basicVector().y(); }),
            pf::rechit::Corner1xBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[1].basicVector().x(); }),
            pf::rechit::Corner1yBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[1].basicVector().y(); }),
            pf::rechit::Corner2xBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[2].basicVector().x(); }),
            pf::rechit::Corner2yBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[2].basicVector().y(); }),
            pf::rechit::Corner3xBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[3].basicVector().x(); }),
            pf::rechit::Corner3yBV::filler(
                [](reco::PFRecHitFraction const* x) { return x->recHitRef()->getCornersXYZ()[3].basicVector().y(); })

                )};
  }

  SuperClusterRecHitTable makeSuperClusterRecHitTable(std::vector<const std::pair<DetId, float>*> const& objects) {
    return {objects,
            edm::soa::column_fillers(
                pf::rechit::DetIdValue::filler([](const std::pair<DetId, float>* x) { return x->first; }),
                pf::rechit::Fraction::filler([](const std::pair<DetId, float>* x) { return x->second; }))};
  }

  ClusterTable makeClusterTable(std::vector<const reco::PFBlockElementCluster*> const& objects) {
    return {objects,
            edm::soa::column_fillers(pf::cluster::Eta::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->positionREP().eta();
                                     }),
                                     pf::cluster::Phi::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->positionREP().phi();
                                     }),
                                     pf::cluster::Posx::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->position().x();
                                     }),
                                     pf::cluster::Posy::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->position().y();
                                     }),
                                     pf::cluster::Posz::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->position().z();
                                     }),
                                     pf::cluster::FracsNbr::filler([](const reco::PFBlockElementCluster* x) {
                                       return x->clusterRef()->recHitFractions().size();
                                     }),
                                     pf::cluster::Layer::filler(
                                         [](const reco::PFBlockElementCluster* x) { return x->clusterRef()->layer(); }),
                                     pf::cluster::SCRefKey::filler([](const reco::PFBlockElementCluster* x) {
                                       return refToElementID(x->superClusterRef());
                                     }))};
  }

  SuperClusterTable makeSuperClusterTable(std::vector<const reco::PFBlockElementSuperCluster*> const& objects) {
    return {objects,
            edm::soa::column_fillers(pf::cluster::Eta::filler([](const reco::PFBlockElementSuperCluster* x) {
                                       return x->superClusterRef()->position().eta();
                                     }),
                                     pf::cluster::Phi::filler([](const reco::PFBlockElementSuperCluster* x) {
                                       return x->superClusterRef()->position().phi();
                                     }),
                                     pf::cluster::SCRefKey::filler([](const reco::PFBlockElementSuperCluster* x) {
                                       return refToElementID(x->superClusterRef());
                                     }))};
  }

  GSFTable makeGSFTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects) {
    return {
        objects,
        edm::soa::column_fillers(
            pf::track::Pt::filler([](const reco::PFBlockElementGsfTrack* x) {
              return sqrt(
                  x->GsftrackPF().extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Vect().Perp2());
            }),
            pf::track::GsfTrackRefPFIsNonNull::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return x->GsftrackRefPF().isNonnull(); }),
            pf::track::GsfTrackRefPFKey::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return refToElementID(x->GsftrackRefPF()); }),
            pf::track::TrackType_FROM_GAMMACONV::filler([](const reco::PFBlockElementGsfTrack* x) {
              return x->trackType(reco::PFBlockElement::T_FROM_GAMMACONV);
            }),
            pf::track::GsfTrackRefPFTrackId::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return x->GsftrackRefPF()->trackId(); }))};
  }
}  // namespace edm::soa