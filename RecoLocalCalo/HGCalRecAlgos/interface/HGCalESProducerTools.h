#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalESProducerTools_h
#define RecoLocalCalo_HGCalRecAlgos_HGCalESProducerTools_h

#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;  // ordered_json preserves key insertion order

namespace hgcal {

  std::string search_modkey(const std::string& module, const json& data, const std::string& name);
  std::string search_fedkey(const int& fedid, const json& data, const std::string& name);
  bool check_keys(const json& data,
                  const std::string& firstkey,
                  const std::vector<std::string>& keys,
                  const std::string& fname);

}  // namespace hgcal

#endif
