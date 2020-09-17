#ifndef RecoParticleFlow_PFProducer_TableDefinitions_h
#define RecoParticleFlow_PFProducer_TableDefinitions_h
#include <set>
#include <vector>

#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"
#include "FWCore/SOA/interface/Column.h"

#include "DataFormats/Provenance/interface/ElementID.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

template <class C>
edm::ElementID refToElementID(const edm::Ref<C>& ref) {
  return edm::ElementID(ref.id(), ref.index());
}

namespace edm::soa {
  namespace col::pf {
    namespace track {
      SOA_DECLARE_COLUMN(Pt, double, "Pt");
      SOA_DECLARE_COLUMN(ExtrapolationValid, bool, "ExtrapolationValid");
      SOA_DECLARE_COLUMN(Eta, double, "Eta");
      SOA_DECLARE_COLUMN(Phi, double, "Phi");
      SOA_DECLARE_COLUMN(Posx, double, "Posx");
      SOA_DECLARE_COLUMN(Posy, double, "Posy");
      SOA_DECLARE_COLUMN(Posz, double, "Posz");
      SOA_DECLARE_COLUMN(PosR, double, "PosR");
      SOA_DECLARE_COLUMN(IsLinkedToDisplacedVertex, bool, "IsLinkedToDisplacedVertex");
      SOA_DECLARE_COLUMN(GsfTrackRefPFKey, edm::ElementID, "GsfTrackRefPFKey");
      SOA_DECLARE_COLUMN(GsfTrackRefPFIsNonNull, bool, "GsfTrackRefPFIsNonNull");
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

      SOA_DECLARE_COLUMN(Corner0x, float, "Corner0x");
      SOA_DECLARE_COLUMN(Corner0y, float, "Corner0y");
      SOA_DECLARE_COLUMN(Corner0z, float, "Corner0z");
      SOA_DECLARE_COLUMN(Corner1x, float, "Corner1x");
      SOA_DECLARE_COLUMN(Corner1y, float, "Corner1y");
      SOA_DECLARE_COLUMN(Corner1z, float, "Corner1z");
      SOA_DECLARE_COLUMN(Corner2x, float, "Corner2x");
      SOA_DECLARE_COLUMN(Corner2y, float, "Corner2y");
      SOA_DECLARE_COLUMN(Corner2z, float, "Corner2z");
      SOA_DECLARE_COLUMN(Corner3x, float, "Corner3x");
      SOA_DECLARE_COLUMN(Corner3y, float, "Corner3y");
      SOA_DECLARE_COLUMN(Corner3z, float, "Corner3z");

      SOA_DECLARE_COLUMN(Corner0xBV, double, "Corner0xBV");
      SOA_DECLARE_COLUMN(Corner0yBV, double, "Corner0yBV");
      SOA_DECLARE_COLUMN(Corner1xBV, double, "Corner1xBV");
      SOA_DECLARE_COLUMN(Corner1yBV, double, "Corner1yBV");
      SOA_DECLARE_COLUMN(Corner2xBV, double, "Corner2xBV");
      SOA_DECLARE_COLUMN(Corner2yBV, double, "Corner2yBV");
      SOA_DECLARE_COLUMN(Corner3xBV, double, "Corner3xBV");
      SOA_DECLARE_COLUMN(Corner3yBV, double, "Corner3yBV");

      SOA_DECLARE_COLUMN(Corner0eta, float, "Corner0eta");
      SOA_DECLARE_COLUMN(Corner0phi, float, "Corner0phi");
      SOA_DECLARE_COLUMN(Corner1eta, float, "Corner1eta");
      SOA_DECLARE_COLUMN(Corner1phi, float, "Corner1phi");
      SOA_DECLARE_COLUMN(Corner2eta, float, "Corner2eta");
      SOA_DECLARE_COLUMN(Corner2phi, float, "Corner2phi");
      SOA_DECLARE_COLUMN(Corner3eta, float, "Corner3eta");
      SOA_DECLARE_COLUMN(Corner3phi, float, "Corner3phi");
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
                                 col::pf::track::DisplacedVertexRef_TO_DISP_IsNonNull,
                                 col::pf::track::DisplacedVertexRef_TO_DISP_Key,
                                 col::pf::track::DisplacedVertexRef_FROM_DISP_IsNonNull,
                                 col::pf::track::DisplacedVertexRef_FROM_DISP_Key>;
  using ConvRefTable = Table<col::pf::track::ConvRefIsNonNull, col::pf::track::ConvRefKey>;

  using GSFTable = Table<col::pf::track::Pt,
                         col::pf::track::GsfTrackRefPFIsNonNull,
                         col::pf::track::GsfTrackRefPFKey,
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
                            col::pf::rechit::Corner0x,
                            col::pf::rechit::Corner0y,
                            col::pf::rechit::Corner0z,
                            col::pf::rechit::Corner1x,
                            col::pf::rechit::Corner1y,
                            col::pf::rechit::Corner1z,
                            col::pf::rechit::Corner2x,
                            col::pf::rechit::Corner2y,
                            col::pf::rechit::Corner2z,
                            col::pf::rechit::Corner3x,
                            col::pf::rechit::Corner3y,
                            col::pf::rechit::Corner3z,
                            col::pf::rechit::Corner0eta,
                            col::pf::rechit::Corner0phi,
                            col::pf::rechit::Corner1eta,
                            col::pf::rechit::Corner1phi,
                            col::pf::rechit::Corner2eta,
                            col::pf::rechit::Corner2phi,
                            col::pf::rechit::Corner3eta,
                            col::pf::rechit::Corner3phi,
                            col::pf::rechit::Corner0xBV,
                            col::pf::rechit::Corner0yBV,
                            col::pf::rechit::Corner1xBV,
                            col::pf::rechit::Corner1yBV,
                            col::pf::rechit::Corner2xBV,
                            col::pf::rechit::Corner2yBV,
                            col::pf::rechit::Corner3xBV,
                            col::pf::rechit::Corner3yBV>;
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
