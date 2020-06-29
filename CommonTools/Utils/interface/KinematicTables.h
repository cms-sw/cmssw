#ifndef CommonTools_Utils_KinematicTables_h
#define CommonTools_Utils_KinematicTables_h

#include "CommonTools/Utils/interface/KinematicColumns.h"
#include "CommonTools/Utils/interface/LazyResult.h"
#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"

#include <vector>

namespace edm {

  namespace soa {

    using EtaPhiTable = edm::soa::Table<col::Eta, col::Phi>;
    using EtaPhiTableView = edm::soa::TableView<col::Eta, col::Phi>;
    using PtEtaPhiTable = edm::soa::Table<col::Pt, col::Eta, col::Phi>;

    template <class Object>
    EtaPhiTable makeEtaPhiTable(std::vector<Object> const& objects) {
      return {objects,
              edm::soa::column_fillers(col::Eta::filler([](Object const& x) { return x.eta(); }),
                                       col::Phi::filler([](Object const& x) { return x.phi(); }))};
    }

    template <class Object>
    auto makeEtaPhiTableLazy(std::vector<Object> const& objects) {
      return LazyResult(&makeEtaPhiTable<Object>, objects);
    }

    template <class Object>
    PtEtaPhiTable makePtEtaPhiTable(std::vector<Object> const& objects) {
      return {objects,
              edm::soa::column_fillers(col::Pt::filler([](Object const& x) { return x.pt(); }),
                                       col::Eta::filler([](Object const& x) { return x.eta(); }),
                                       col::Phi::filler([](Object const& x) { return x.phi(); }))};
    using TrackTable = edm::soa::Table<col::Px,
                                       col::Py,
                                       col::Pz,
                                       col::P,
                                       col::PtError,
                                       col::MissingInnerHits,
                                       col::NumberOfValidHits,
                                       col::Charge,
                                       col::Eta,
                                       col::Phi,
                                       col::Pt,
                                       col::D0>;
    using TrackTableView = edm::soa::TableView<col::Px,
                                               col::Py,
                                               col::Pz,
                                               col::P,
                                               col::PtError,
                                               col::MissingInnerHits,
                                               col::NumberOfValidHits,
                                               col::Charge,
                                               col::Eta,
                                               col::Phi,
                                               col::Pt,
                                               col::D0>;

    template <class Object>
    TrackTable makeTrackTable(std::vector<Object> const& objects) {
      return {objects,
              edm::soa::column_fillers(
                  col::D0::filler([](Object const& x) { return x.d0(); }),
                  col::Px::filler([](Object const& x) { return x.px(); }),
                  col::Py::filler([](Object const& x) { return x.py(); }),
                  col::Pz::filler([](Object const& x) { return x.pz(); }),
                  col::P::filler([](Object const& x) { return x.p(); }),
                  col::Pt::filler([](Object const& x) { return x.pt(); }),
                  col::PtError::filler([](Object const& x) { return x.ptError(); }),
                  col::MissingInnerHits::filler([](Object const& x) {
                    return x.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
                  }),
                  col::NumberOfValidHits::filler([](Object const& x) { return x.numberOfValidHits(); }),
                  col::Charge::filler([](Object const& x) { return x.charge(); }),
                  col::Eta::filler([](Object const& x) { return x.eta(); }),
                  col::Phi::filler([](Object const& x) { return x.phi(); }))};
    }

    template <class Object>
    auto makeTrackTableLazy(std::vector<Object> const& objects) {
      return LazyResult(&makeTrackTable<Object>, objects);
    }

  }  // namespace soa

}  // namespace edm

#endif
