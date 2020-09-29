#include "RecoParticleFlow/PFProducer/interface/PFTables.h"

using namespace edm::soa::col;
namespace edm::soa {

  TrackTableVertex makeTrackTableVertex(std::vector<reco::PFBlockElement*> const& objects) {
    using namespace edm::soa::col::pf::track;
    return {
        objects,
        edm::soa::column_fillers(
            ExtrapolationValid::filler([](reco::PFBlockElement* x) {
              return x->trackRefPF()->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).isValid();
            }),
            Pt::filler([](reco::PFBlockElement* x) {
              return sqrt(
                  x->trackRefPF()->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Vect().Perp2());
            }),
            IsLinkedToDisplacedVertex::filler([](reco::PFBlockElement* x) { return x->isLinkedToDisplacedVertex(); }),
            TrackType_FROM_GAMMACONV::filler(
                [](reco::PFBlockElement* x) { return x->trackType(reco::PFBlockElement::T_FROM_GAMMACONV); }),
            TrackType_FROM_V0::filler(
                [](reco::PFBlockElement* x) { return x->trackType(reco::PFBlockElement::T_FROM_V0); }),
            V0RefIsNonNull::filler([](reco::PFBlockElement* x) { return x->V0Ref().isNonnull(); }),
            V0RefKey::filler([](reco::PFBlockElement* x) { return edm::refToElementID(x->V0Ref()); }),
            KfTrackRefIsNonNull::filler([](reco::PFBlockElement* x) { return x->trackRefPF()->trackRef().isNonnull(); }),
            KfTrackRefKey::filler(
                [](reco::PFBlockElement* x) { return edm::refToElementID(x->trackRefPF()->trackRef()); }),
            KfTrackRefBaseKey::filler([](reco::PFBlockElement* x) {
              return edm::refToElementID(reco::TrackBaseRef(x->trackRefPF()->trackRef()));
            }),
            DisplacedVertexRef_TO_DISP_IsNonNull::filler([](reco::PFBlockElement* x) {
              return x->displacedVertexRef(reco::PFBlockElement::T_TO_DISP).isNonnull();
            }),
            DisplacedVertexRef_TO_DISP_Key::filler([](reco::PFBlockElement* x) {
              return edm::refToElementID(x->displacedVertexRef(reco::PFBlockElement::T_TO_DISP));
            }),
            DisplacedVertexRef_FROM_DISP_IsNonNull::filler([](reco::PFBlockElement* x) {
              return x->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP).isNonnull();
            }),
            DisplacedVertexRef_FROM_DISP_Key::filler([](reco::PFBlockElement* x) {
              return edm::refToElementID(x->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP));
            }))};
  }

  ConvRefTable makeConvRefTable(const std::vector<reco::ConversionRef>& convrefs) {
    return {convrefs,
            edm::soa::column_fillers(
                pf::track::ConvRefIsNonNull::filler([](const reco::ConversionRef& x) { return x.isNonnull(); }),
                pf::track::ConvRefKey::filler([](const reco::ConversionRef& x) { return edm::refToElementID(x); }))};
  }

  ConvBremTable makeConvBremTable(const std::vector<reco::PFRecTrackRef>& convbrems) {
    return {convbrems,
            edm::soa::column_fillers(pf::track::ConvBremRefKey::filler(
                                         [](reco::PFRecTrackRef x) { return edm::refToElementID(x->trackRef()); }),
                                     pf::track::ConvBremRefBaseKey::filler([](reco::PFRecTrackRef x) {
                                       return edm::refToElementID(reco::TrackBaseRef(x->trackRef()));
                                     }))};
  }

  BremTable makeBremTable(const std::vector<const reco::PFBlockElementBrem*>& brems) {
    return {
        brems,
        edm::soa::column_fillers(
            pf::track::Pt::filler([](const reco::PFBlockElementBrem* x) {
              return sqrt(
                  x->trackPF().extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Vect().Perp2());
            }),
            pf::track::GsfTrackRefPFIsNonNull::filler(
                [](const reco::PFBlockElementBrem* x) { return x->GsftrackRefPF().isNonnull(); }),
            pf::track::GsfTrackRefPFKey::filler(
                [](const reco::PFBlockElementBrem* x) { return edm::refToElementID(x->GsftrackRefPF()); }))};
  }

  TrackTableExtrapolation makeTrackTable(std::vector<reco::PFBlockElement*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    std::vector<reco::PFRecTrackRef> rectracks;
    rectracks.reserve(objects.size());
    for (const auto& obj : objects) {
      rectracks.push_back(obj->trackRefPF());
    }
    return makeTrackTable(rectracks, layerType);
  }

  //this should be unified with the function above
  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    std::vector<const reco::GsfPFRecTrack*> rectracks;
    rectracks.reserve(objects.size());
    for (const auto* obj : objects) {
      rectracks.push_back(&(obj->GsftrackPF()));
    }
    return makeTrackTable(rectracks, layerType);
  }

  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementBrem*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    std::vector<const reco::PFRecTrack*> rectracks;
    rectracks.reserve(objects.size());
    for (const auto* obj : objects) {
      rectracks.push_back(&(obj->trackPF()));
    }
    return makeTrackTable(rectracks, layerType);
  }

  template <class RecTrackType>
  TrackTableExtrapolation makeTrackTable(std::vector<RecTrackType> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType) {
    using namespace edm::soa::col::pf::track;
    return {
        objects,
        edm::soa::column_fillers(
            ExtrapolationValid::filler(
                [layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).isValid(); }),
            Eta::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().eta(); }),
            Phi::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().phi(); }),
            Posx::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().X(); }),
            Posy::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().Y(); }),
            Posz::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().Z(); }),
            PosR::filler([layerType](RecTrackType x) { return x->extrapolatedPoint(layerType).positionREP().R(); }))};
  }

  RecHitTable makeRecHitTable(std::vector<const reco::PFRecHitFraction*> const& objects) {
    using namespace edm::soa::col::pf::rechit;
    return {objects,
            edm::soa::column_fillers(
                DetIdValue::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->detId(); }),
                Fraction::filler([](reco::PFRecHitFraction const* x) { return x->fraction(); }),
                Eta::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->positionREP().eta(); }),
                Phi::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->positionREP().phi(); }),
                Posx::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().x(); }),
                Posy::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().y(); }),
                Posz::filler([](reco::PFRecHitFraction const* x) { return x->recHitRef()->position().z(); }),
                CornerX::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsF {
                  return {{x->recHitRef()->getCornersXYZ()[0].x(),
                           x->recHitRef()->getCornersXYZ()[1].x(),
                           x->recHitRef()->getCornersXYZ()[2].x(),
                           x->recHitRef()->getCornersXYZ()[3].x()}};
                }),
                CornerY::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsF {
                  return {{x->recHitRef()->getCornersXYZ()[0].y(),
                           x->recHitRef()->getCornersXYZ()[1].y(),
                           x->recHitRef()->getCornersXYZ()[2].y(),
                           x->recHitRef()->getCornersXYZ()[3].y()}};
                }),
                CornerZ::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsF {
                  return {{x->recHitRef()->getCornersXYZ()[0].z(),
                           x->recHitRef()->getCornersXYZ()[1].z(),
                           x->recHitRef()->getCornersXYZ()[2].z(),
                           x->recHitRef()->getCornersXYZ()[3].z()}};
                }),
                CornerEta::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsF {
                  return {{x->recHitRef()->getCornersREP()[0].eta(),
                           x->recHitRef()->getCornersREP()[1].eta(),
                           x->recHitRef()->getCornersREP()[2].eta(),
                           x->recHitRef()->getCornersREP()[3].eta()}};
                }),
                CornerPhi::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsF {
                  return {{x->recHitRef()->getCornersREP()[0].phi(),
                           x->recHitRef()->getCornersREP()[1].phi(),
                           x->recHitRef()->getCornersREP()[2].phi(),
                           x->recHitRef()->getCornersREP()[3].phi()}};
                }),
                CornerXBV::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsD {
                  return {{x->recHitRef()->getCornersXYZ()[0].basicVector().x(),
                           x->recHitRef()->getCornersXYZ()[1].basicVector().x(),
                           x->recHitRef()->getCornersXYZ()[2].basicVector().x(),
                           x->recHitRef()->getCornersXYZ()[3].basicVector().x()}};
                }),
                CornerYBV::filler([](reco::PFRecHitFraction const* x) -> col::pf::CornerCoordsD {
                  return {{x->recHitRef()->getCornersXYZ()[0].basicVector().y(),
                           x->recHitRef()->getCornersXYZ()[1].basicVector().y(),
                           x->recHitRef()->getCornersXYZ()[2].basicVector().y(),
                           x->recHitRef()->getCornersXYZ()[3].basicVector().y()}};
                })

                    )};
  }

  SuperClusterRecHitTable makeSuperClusterRecHitTable(std::vector<const std::pair<DetId, float>*> const& objects) {
    using namespace edm::soa::col::pf::rechit;
    return {objects,
            edm::soa::column_fillers(DetIdValue::filler([](const std::pair<DetId, float>* x) { return x->first; }),
                                     Fraction::filler([](const std::pair<DetId, float>* x) { return x->second; }))};
  }

  ClusterTable makeClusterTable(std::vector<const reco::PFBlockElementCluster*> const& objects) {
    using namespace edm::soa::col::pf::cluster;
    return {objects,
            edm::soa::column_fillers(
                Eta::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->positionREP().eta(); }),
                Phi::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->positionREP().phi(); }),
                Posx::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->position().x(); }),
                Posy::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->position().y(); }),
                Posz::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->position().z(); }),
                FracsNbr::filler(
                    [](const reco::PFBlockElementCluster* x) { return x->clusterRef()->recHitFractions().size(); }),
                Layer::filler([](const reco::PFBlockElementCluster* x) { return x->clusterRef()->layer(); }),
                SCRefIsNonNull::filler(
                    [](const reco::PFBlockElementCluster* x) { return x->superClusterRef().isNonnull(); }),
                SCRefKey::filler(
                    [](const reco::PFBlockElementCluster* x) { return edm::refToElementID(x->superClusterRef()); }))};
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
                                       return edm::refToElementID(x->superClusterRef());
                                     }))};
  }

  GSFTable makeGSFTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects) {
    using namespace edm::soa::col::pf::track;

    return {
        objects,
        edm::soa::column_fillers(
            Pt::filler([](const reco::PFBlockElementGsfTrack* x) {
              return sqrt(
                  x->GsftrackPF().extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Vect().Perp2());
            }),
            GsfTrackRefPFIsNonNull::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return x->GsftrackRefPF().isNonnull(); }),
            GsfTrackRefPFKey::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return edm::refToElementID(x->GsftrackRefPF()); }),
            KfPFRecTrackRefIsNonNull::filler([](const reco::PFBlockElementGsfTrack* x) {
              return x->GsftrackRefPF()->kfPFRecTrackRef().isNonnull();
            }),
            KfPFRecTrackRefKey::filler([](const reco::PFBlockElementGsfTrack* x) {
              return edm::refToElementID(x->GsftrackRefPF()->kfPFRecTrackRef());
            }),
            KfTrackRefIsNonNull::filler([](const reco::PFBlockElementGsfTrack* x) {
              const auto& r1 = x->GsftrackRefPF();
              bool ret = false;
              if (r1.isNonnull()) {
                const auto r2 = r1->kfPFRecTrackRef();
                if (r2.isNonnull()) {
                  const auto r3 = r2->trackRef();
                  ret = r3.isNonnull();
                }
              }
              return ret;
            }),
            KfTrackRefKey::filler([](const reco::PFBlockElementGsfTrack* x) {
              const auto& r1 = x->GsftrackRefPF();
              edm::ElementID ret;
              if (r1.isNonnull()) {
                const auto r2 = r1->kfPFRecTrackRef();
                if (r2.isNonnull()) {
                  const auto r3 = r2->trackRef();
                  ret = edm::refToElementID(r3);
                }
              }
              return ret;
            }),
            TrackType_FROM_GAMMACONV::filler([](const reco::PFBlockElementGsfTrack* x) {
              return x->trackType(reco::PFBlockElement::T_FROM_GAMMACONV);
            }),
            GsfTrackRefPFTrackId::filler(
                [](const reco::PFBlockElementGsfTrack* x) { return x->GsftrackRefPF()->trackId(); }))};
  }
}  // namespace edm::soa