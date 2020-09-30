#ifndef RecoParticleFlow_PFProducer_TableDefinitions_h
#define RecoParticleFlow_PFProducer_TableDefinitions_h
#include <set>
#include <vector>

#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/Column.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/Common/interface/refToElementID.h"

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
      SOA_DECLARE_COLUMN(SCRefIsNonNull, bool, "SCRefIsNonNull");
      SOA_DECLARE_COLUMN(SCRefKey, edm::ElementID, "SCRefKey");
    }  // namespace cluster
  }    // namespace col::pf
}  // namespace edm::soa

//define namespace aliases, but avoid leaking them to other files
namespace {
  namespace cluster = edm::soa::col::pf::cluster;
  namespace track = edm::soa::col::pf::track;
  namespace rechit = edm::soa::col::pf::rechit;
};  // namespace

namespace edm::soa {
  using TrackTableVertex = Table<track::ExtrapolationValid,
                                 track::Pt,
                                 track::IsLinkedToDisplacedVertex,
                                 track::TrackType_FROM_GAMMACONV,
                                 track::TrackType_FROM_V0,
                                 track::V0RefIsNonNull,
                                 track::V0RefKey,
                                 track::KfTrackRefIsNonNull,
                                 track::KfTrackRefKey,
                                 track::KfTrackRefBaseKey,
                                 track::DisplacedVertexRef_TO_DISP_IsNonNull,
                                 track::DisplacedVertexRef_TO_DISP_Key,
                                 track::DisplacedVertexRef_FROM_DISP_IsNonNull,
                                 track::DisplacedVertexRef_FROM_DISP_Key>;
  using BremTable = Table<track::Pt, track::GsfTrackRefPFIsNonNull, track::GsfTrackRefPFKey>;

  using ConvRefTable = Table<track::ConvRefIsNonNull, track::ConvRefKey>;
  using ConvBremTable = Table<track::ConvBremRefKey, track::ConvBremRefBaseKey>;

  using GSFTable = Table<track::Pt,
                         track::GsfTrackRefPFIsNonNull,
                         track::GsfTrackRefPFKey,
                         track::KfPFRecTrackRefIsNonNull,
                         track::KfPFRecTrackRefKey,
                         track::KfTrackRefIsNonNull,
                         track::KfTrackRefKey,
                         track::TrackType_FROM_GAMMACONV,
                         track::GsfTrackRefPFTrackId>;
  using TrackTableExtrapolation =
      Table<track::ExtrapolationValid, track::Eta, track::Phi, track::Posx, track::Posy, track::Posz, track::PosR>;
  using RecHitTable = Table<rechit::DetIdValue,
                            rechit::Fraction,
                            rechit::Eta,
                            rechit::Phi,
                            rechit::Posx,
                            rechit::Posy,
                            rechit::Posz,
                            rechit::CornerX,
                            rechit::CornerY,
                            rechit::CornerZ,
                            rechit::CornerEta,
                            rechit::CornerPhi,
                            rechit::CornerXBV,
                            rechit::CornerYBV>;
  using ClusterTable = Table<cluster::Eta,
                             cluster::Phi,
                             cluster::Posx,
                             cluster::Posy,
                             cluster::Posz,
                             cluster::FracsNbr,
                             cluster::Layer,
                             cluster::SCRefIsNonNull,
                             cluster::SCRefKey>;
  using SuperClusterTable = Table<cluster::Eta, cluster::Phi, cluster::SCRefKey>;
  using SuperClusterRecHitTable = Table<rechit::DetIdValue, rechit::Fraction>;
}  // namespace edm::soa

#endif
