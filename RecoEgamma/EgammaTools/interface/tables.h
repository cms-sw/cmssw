#ifndef RecoEgamma_EgammaTools_tables_h
#define RecoEgamma_EgammaTools_tables_h

#include "FWCore/SOA/interface/Column.h"
#include "FWCore/SOA/interface/Table.h"
#include "RecoEgamma/EgammaTools/interface/LazyResult.h"

#include <vector>

namespace egamma {

  namespace soa {

    namespace col {

      SOA_DECLARE_COLUMN(Pt, float, "pt");
      SOA_DECLARE_COLUMN(Eta, float, "eta");
      SOA_DECLARE_COLUMN(Phi, float, "phi");
      SOA_DECLARE_COLUMN(Vz, float, "vz");

    }  // namespace col

    using EtaPhiTable = edm::soa::Table<col::Eta, col::Phi>;

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

  }  // namespace soa

}  // namespace egamma

#endif
