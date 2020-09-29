#ifndef RecoParticleFlow_PFProducer_TableDefinitions_h
#define RecoParticleFlow_PFProducer_TableDefinitions_h
#include <set>
#include <vector>

#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/Column.h"

#include "DataFormats/Provenance/interface/ElementID.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

template <class C>
edm::ElementID refToElementID(const edm::Ref<C>& ref) {
  return edm::ElementID(ref.id(), ref.index());
}

template <class C>
edm::ElementID refToElementID(const edm::RefToBase<C>& ref) {
  return edm::ElementID(ref.id(), ref.key());
}
namespace edm::soa {
  namespace col::pf {
    using CornerCoordsD = std::array<double, 4>;
    using CornerCoordsF = std::array<float, 4>;

    namespace track {
      SOA_DECLARE_COLUMN(Pt, double, "Pt");
      SOA_DECLARE_COLUMN(ExtrapolationValid, bool, "ExtrapolationValid");
      SOA_DECLARE_COLUMN(Eta, double, "Eta");
      SOA_DECLARE_COLUMN(Phi, double, "Phi");
      SOA_DECLARE_COLUMN(Posx, double, "Posx");
      SOA_DECLARE_COLUMN(Posy, double, "Posy");
      SOA_DECLARE_COLUMN(Posz, double, "Posz");
      SOA_DECLARE_COLUMN(PosR, double, "PosR");
      SOA_DECLARE_COLUMN(ConvBremRefKey, edm::ElementID, "ConvBremRefKey");
      SOA_DECLARE_COLUMN(ConvBremRefBaseKey, edm::ElementID, "ConvBremRefBaseKey");
      SOA_DECLARE_COLUMN(IsLinkedToDisplacedVertex, bool, "IsLinkedToDisplacedVertex");
      SOA_DECLARE_COLUMN(GsfTrackRefPFKey, edm::ElementID, "GsfTrackRefPFKey");
      SOA_DECLARE_COLUMN(GsfTrackRefPFIsNonNull, bool, "GsfTrackRefPFIsNonNull");
      SOA_DECLARE_COLUMN(KfPFRecTrackRefKey, edm::ElementID, "KfPFRecTrackRefKey");
      SOA_DECLARE_COLUMN(KfPFRecTrackRefIsNonNull, bool, "KfPFRecTrackRefIsNonNull");
      SOA_DECLARE_COLUMN(KfTrackRefKey, edm::ElementID, "KfTrackRefKey");
      SOA_DECLARE_COLUMN(KfTrackRefBaseKey, edm::ElementID, "KfTrackRefBaseKey");
      SOA_DECLARE_COLUMN(KfTrackRefIsNonNull, bool, "KfTrackRefIsNonNull");
      SOA_DECLARE_COLUMN(ConvRefKey, edm::ElementID, "ConvRefKey");
      SOA_DECLARE_COLUMN(ConvRefIsNonNull, bool, "ConvRefIsNonNull");
      SOA_DECLARE_COLUMN(V0RefKey, edm::ElementID, "V0RefKey");
      SOA_DECLARE_COLUMN(V0RefIsNonNull, bool, "V0RefIsNonNull");
      SOA_DECLARE_COLUMN(TrackType_FROM_GAMMACONV, bool, "TrackType_FROM_GAMMACONV");
      SOA_DECLARE_COLUMN(TrackType_FROM_V0, bool, "TrackType_FROM_V0");
      SOA_DECLARE_COLUMN(GsfTrackRefPFTrackId, int, "GsfTrackRefPFTrackId");
      SOA_DECLARE_COLUMN(DisplacedVertexRef_TO_DISP_IsNonNull, bool, "DisplacedVertexRef_TO_DISP_IsNonNull");
      SOA_DECLARE_COLUMN(DisplacedVertexRef_TO_DISP_Key, edm::ElementID, "DisplacedVertexRef_TO_DISP_Key");
      SOA_DECLARE_COLUMN(DisplacedVertexRef_FROM_DISP_IsNonNull, bool, "DisplacedVertexRef_FROM_DISP_IsNonNull");
      SOA_DECLARE_COLUMN(DisplacedVertexRef_FROM_DISP_Key, edm::ElementID, "DisplacedVertexRef_FROM_DISP_Key");
    };  // namespace track
    namespace rechit {
      SOA_DECLARE_COLUMN(DetIdValue, unsigned int, "DetIdValue");
      SOA_DECLARE_COLUMN(Fraction, double, "Fraction");
      SOA_DECLARE_COLUMN(Eta, double, "Eta");
      SOA_DECLARE_COLUMN(Phi, double, "Phi");
      SOA_DECLARE_COLUMN(Posx, double, "Posx");
      SOA_DECLARE_COLUMN(Posy, double, "Posy");
      SOA_DECLARE_COLUMN(Posz, double, "Posz");

      SOA_DECLARE_COLUMN(CornerX, CornerCoordsF, "CornerX");
      SOA_DECLARE_COLUMN(CornerY, CornerCoordsF, "CornerY");
      SOA_DECLARE_COLUMN(CornerZ, CornerCoordsF, "CornerZ");

      SOA_DECLARE_COLUMN(CornerXBV, CornerCoordsD, "CornerXBV");
      SOA_DECLARE_COLUMN(CornerYBV, CornerCoordsD, "CornerYBV");

      SOA_DECLARE_COLUMN(CornerEta, CornerCoordsF, "CornerEta");
      SOA_DECLARE_COLUMN(CornerPhi, CornerCoordsF, "CornerPhi");
    }  // namespace rechit
    namespace cluster {
      SOA_DECLARE_COLUMN(Eta, double, "Eta");
      SOA_DECLARE_COLUMN(Phi, double, "Phi");
      SOA_DECLARE_COLUMN(Posx, double, "Posx");
      SOA_DECLARE_COLUMN(Posy, double, "Posy");
      SOA_DECLARE_COLUMN(Posz, double, "Posz");
      SOA_DECLARE_COLUMN(FracsNbr, int, "FracsNbr");
      SOA_DECLARE_COLUMN(Layer, PFLayer::Layer, "Layer");
      SOA_DECLARE_COLUMN(SCRefKey, edm::ElementID, "SCRefKey");
    }  // namespace cluster
  }    // namespace col::pf
  using TrackTableVertex = Table<col::pf::track::ExtrapolationValid,
                                 col::pf::track::Pt,
                                 col::pf::track::IsLinkedToDisplacedVertex,
                                 col::pf::track::TrackType_FROM_GAMMACONV,
                                 col::pf::track::TrackType_FROM_V0,
                                 col::pf::track::V0RefIsNonNull,
                                 col::pf::track::V0RefKey,
                                 col::pf::track::KfTrackRefIsNonNull,
                                 col::pf::track::KfTrackRefKey,
                                 col::pf::track::KfTrackRefBaseKey,
                                 col::pf::track::DisplacedVertexRef_TO_DISP_IsNonNull,
                                 col::pf::track::DisplacedVertexRef_TO_DISP_Key,
                                 col::pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull,
                                 col::pf::track::DisplacedVertexRef_FROM_DISP_Key>;
  using BremTable = Table<col::pf::track::Pt, col::pf::track::GsfTrackRefPFIsNonNull, col::pf::track::GsfTrackRefPFKey>;

  using ConvRefTable = Table<col::pf::track::ConvRefIsNonNull, col::pf::track::ConvRefKey>;
  using ConvBremTable = Table<col::pf::track::ConvBremRefKey, col::pf::track::ConvBremRefBaseKey>;

  using GSFTable = Table<col::pf::track::Pt,
                         col::pf::track::GsfTrackRefPFIsNonNull,
                         col::pf::track::GsfTrackRefPFKey,
                         col::pf::track::KfPFRecTrackRefIsNonNull,
                         col::pf::track::KfPFRecTrackRefKey,
                         col::pf::track::KfTrackRefIsNonNull,
                         col::pf::track::KfTrackRefKey,
                         col::pf::track::TrackType_FROM_GAMMACONV,
                         col::pf::track::GsfTrackRefPFTrackId>;
  using TrackTableExtrapolation = Table<col::pf::track::ExtrapolationValid,
                                        col::pf::track::Eta,
                                        col::pf::track::Phi,
                                        col::pf::track::Posx,
                                        col::pf::track::Posy,
                                        col::pf::track::Posz,
                                        col::pf::track::PosR>;
  using RecHitTable = Table<col::pf::rechit::DetIdValue,
                            col::pf::rechit::Fraction,
                            col::pf::rechit::Eta,
                            col::pf::rechit::Phi,
                            col::pf::rechit::Posx,
                            col::pf::rechit::Posy,
                            col::pf::rechit::Posz,
                            col::pf::rechit::CornerX,
                            col::pf::rechit::CornerY,
                            col::pf::rechit::CornerZ,
                            col::pf::rechit::CornerEta,
                            col::pf::rechit::CornerPhi,
                            col::pf::rechit::CornerXBV,
                            col::pf::rechit::CornerYBV>;
  using ClusterTable = Table<col::pf::cluster::Eta,
                             col::pf::cluster::Phi,
                             col::pf::cluster::Posx,
                             col::pf::cluster::Posy,
                             col::pf::cluster::Posz,
                             col::pf::cluster::FracsNbr,
                             col::pf::cluster::Layer,
                             col::pf::cluster::SCRefKey>;
  using SuperClusterTable = Table<col::pf::cluster::Eta, col::pf::cluster::Phi, col::pf::cluster::SCRefKey>;
  using SuperClusterRecHitTable = Table<col::pf::rechit::DetIdValue, col::pf::rechit::Fraction>;
}  // namespace edm::soa

#endif
