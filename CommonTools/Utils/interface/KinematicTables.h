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
    using EtaPhiTableView = edm::soa::ViewFromTable_t<EtaPhiTable>;

    using PtEtaPhiTable = edm::soa::Table<col::Pt, col::Eta, col::Phi>;
    using PtEtaPhiTableView = edm::soa::ViewFromTable_t<PtEtaPhiTable>;

    template <class Object>
    EtaPhiTable makeEtaPhiTable(std::vector<Object> const& objects) {
      return {objects};
    }

    template <class Object>
    auto makeEtaPhiTableLazy(std::vector<Object> const& objects) {
      return LazyResult(&makeEtaPhiTable<Object>, objects);
    }

  }  // namespace soa

}  // namespace edm

#endif
