#ifndef FWCore_Services_monitor_file_utilities_h
#define FWCore_Services_monitor_file_utilities_h

#include <iomanip>
#include <iostream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

namespace edm::service::monitor_file_utilities {

  inline auto stream_id(edm::StreamContext const& cs) { return cs.streamID().value(); }

  inline auto module_id(edm::ModuleCallingContext const& mcc) { return mcc.moduleDescription()->id(); }

  inline auto module_id(edm::ESModuleCallingContext const& mcc) { return mcc.componentDescription()->id_; }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value> concatenate(std::ostream& os, T const t) {
    os << ' ' << t;
  }

  template <typename H, typename... T>
  std::enable_if_t<std::is_integral<H>::value> concatenate(std::ostream& os, H const h, T const... t) {
    os << ' ' << h;
    concatenate(os, t...);
  }

  void moduleIdToLabel(std::ostream&,
                       std::vector<std::string> const& iModules,
                       char moduleIdSymbol,
                       std::string const& iIDHeader,
                       std::string const& iLabelHeader);
}  // namespace edm::service::monitor_file_utilities
#endif
