#ifndef CommonTools_Utils_KinematicTables_h
#define CommonTools_Utils_KinematicTables_h

#include "CommonTools/Utils/interface/KinematicColumns.h"
#include "FWCore/SOA/interface/Table.h"
#include "FWCore/SOA/interface/TableView.h"

namespace edm::soa {

  using EtaPhiTable = edm::soa::Table<col::Eta, col::Phi>;
  using EtaPhiTableView = edm::soa::ViewFromTable_t<EtaPhiTable>;

  using PtEtaPhiTable = edm::soa::AddColumns_t<EtaPhiTable, std::tuple<col::Pt>>;
  using PtEtaPhiTableView = edm::soa::ViewFromTable_t<PtEtaPhiTable>;

}  // namespace edm::soa

#endif
