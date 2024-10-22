/**
 * \class GlobalObject
 *
 *
 * Description: define an enumeration of L1 GT objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

#include "DataFormats/L1TGlobal/interface/GlobalObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

l1t::GlobalObject l1t::GlobalObjectStringToEnum(const std::string& label) {
  l1t::GlobalObject ret = l1t::ObjNull;
  unsigned int nMatches = 0;

  for (auto const& [value, name] : l1t::kGlobalObjectEnumStringPairs) {
    if (name == label) {
      ++nMatches;
      ret = value;
    }
  }

  if (nMatches == 0) {
    edm::LogWarning("l1tGlobalObjectStringToEnum")
        << "Failed to find l1t::GlobalObject corresponding to \"" << label << "\"."
        << " Will return l1t::ObjNull (" << ret << ").";
  } else if (nMatches > 1) {
    edm::LogError("l1tGlobalObjectStringToEnum")
        << "Multiple matches (" << nMatches << ") found for label \"" << label << "\"."
        << " Will return last valid match (" << ret << ")."
        << " Please remove duplicates from l1t::kGlobalObjectEnumStringPairs !!";
  }

  return ret;
}

std::string l1t::GlobalObjectEnumToString(const l1t::GlobalObject& gtObject) {
  std::string ret = "ObjNull";
  unsigned int nMatches = 0;

  for (auto const& [value, name] : l1t::kGlobalObjectEnumStringPairs) {
    if (value == gtObject) {
      ++nMatches;
      ret = name;
    }
  }

  if (nMatches == 0) {
    edm::LogWarning("l1TGtObjectEnumToString") << "Failed to find l1t::GlobalObject with a value of " << gtObject << "."
                                               << " Will return \"" << ret << "\".";
  } else if (nMatches > 1) {
    edm::LogError("l1TGtObjectEnumToString")
        << "Multiple matches (" << nMatches << ") found for l1t::GlobalObject value of " << gtObject
        << ". Will return last valid match (\"" << ret << "\")."
        << " Please remove duplicates from l1t::kGlobalObjectEnumStringPairs !!";
  }

  return ret;
}
