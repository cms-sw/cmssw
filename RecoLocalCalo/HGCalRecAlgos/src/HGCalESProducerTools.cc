#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include <sstream>  // for std::istringstream
#include <regex>

namespace hgcal {

  // @short search first match to a given typecode key in a JSON (following insertion order)
  // allow glob patterns, e.g. 'ML-*', 'M[LH]-[A-Z]3*', etc.
  std::string search_modkey(const std::string& module, const json& data, const std::string& name = "") {
    if (!data.is_object() or data.empty()) {
      cms::Exception ex("InvalidData");
      ex << "'" << name << "' does not have the expected map/dict structure!";
      ex.addContext("Calling hgcal::search_modkey()");
      throw ex;
    }
    for (auto it = data.begin(); it != data.end(); ++it) {
      std::regex re(edm::glob2reg(it.key()));
      if (std::regex_match(module, re)) {  // found matching key !
        edm::LogInfo("search_modkey") << "search_modkey: Matched module='" << module << "' to modkey='" << it.key()
                                      << "'";
        return it.key();  // return matching key
      }
    }
    cms::Exception ex("InvalidData");
    ex << "Could not find matching key for '" << module << "' in '" << name << "'! Returning first key '"
       << data.begin().key() << "'...";
    ex.addContext("Calling hgcal::search_modkey()");
    throw ex;
  }

  // @short search first match to a given FED index in a JSON (following insertion order)
  // allow glob patterns like. '1*', '1[0-5]', etc. and
  // allow numerical ranges like '0-20', '20-40', etc.
  std::string search_fedkey(const int& fedid, const json& data, const std::string& name = "") {
    if (!data.is_object() or data.empty()) {
      cms::Exception ex("InvalidData");
      ex << "'" << name << "' does not have the expected map/dict structure!";
      ex.addContext("Calling hgcal::search_fedkey()");
      throw ex;
    }
    auto it = data.begin();
    std::string matchedkey = data.begin().key();  // use first key as default
    while (it != data.end()) {
      std::string fedkey = it.key();

      // try as numerical range
      int low, high;
      char dash;
      std::istringstream iss(fedkey.c_str());
      iss >> low >> dash >> high;              // parse [integer][character][integer] pattern
      if (iss.eof() and dash == '-') {         // matches pattern
        if (low <= fedid and fedid <= high) {  // matches numerical range
          matchedkey = fedkey;
          break;
        }

        // try as glob pattern
      } else {
        const std::string sfedid = std::to_string(fedid);
        std::regex re(edm::glob2reg(fedkey));
        if (std::regex_match(sfedid, re)) {  // found matching key !
          matchedkey = fedkey;
          break;
        }
      }

      ++it;
    }
    if (it == data.end()) {
      cms::Exception ex("InvalidData");
      ex << "Could not find matching key for '" << fedid << "' in '" << name << "'! Returning first key '" << matchedkey
         << "'...";
      ex.addContext("Calling hgcal::search_fedkey()");
      throw ex;
    } else {
      edm::LogInfo("search_fedkey") << "search_fedkey: Matched module='" << fedid << "' to fedkey='" << matchedkey
                                    << "'";
    }

    return matchedkey;  // no matching key found in whole JSON map
  }

  // @short check if JSON data contains key
  bool check_keys(const json& data,
                  const std::string& firstkey,
                  const std::vector<std::string>& keys,
                  const std::string& fname) {
    bool iscomplete = true;
    for (auto const& key : keys) {
      if (not data[firstkey].contains(key)) {
        edm::LogWarning("checkkeys") << " JSON is missing key '" << key << "' for " << firstkey << "!"
                                     << " Please check file " << fname;
        iscomplete = false;
      }
    }
    return iscomplete;
  }

  // @short check if JSON data contains key
  bool check_keys(const json& data, const std::vector<std::string>& keys, const std::string& fname) {
    bool iscomplete = true;
    for (auto const& key : keys) {
      if (not data.contains(key)) {
        edm::LogWarning("checkkeys") << " JSON is missing key '" << key << "'! Please check file " << fname;
        iscomplete = false;
      }
    }
    return iscomplete;
  }

}  // namespace hgcal
