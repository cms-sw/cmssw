#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/Singleton.h"

#include <ext/alloc_traits.h>
#include <cstdlib>
#include <sstream>
#include <mutex>

std::ostream& operator<<(std::ostream& os, const DDName& n) {
  os << n.ns() << ':' << n.name();
  return os;
}

DDName::DDName(const std::string& name, const std::string& ns) : id_(registerName(std::make_pair(name, ns))->second) {}

DDName::DDName(const std::string& name) : id_(0) {
  std::pair<std::string, std::string> result = DDSplit(name);
  if (result.second.empty()) {
    id_ = registerName(std::make_pair(result.first, DDCurrentNamespace::ns()))->second;
  } else {
    id_ = registerName(result)->second;
  }
}

DDName::DDName(const char* name) : id_(0) {
  std::pair<std::string, std::string> result = DDSplit(name);
  if (result.second.empty()) {
    id_ = registerName(std::make_pair(result.first, DDCurrentNamespace::ns()))->second;
  } else {
    id_ = registerName(result)->second;
  }
}

DDName::DDName(const char* name, const char* ns)
    : id_(registerName(std::make_pair(std::string(name), std::string(ns)))->second) {}

DDName::DDName() : id_(0) {}

const std::string& DDName::name() const {
  const static std::string ano_("anonymous");
  const std::string* result;
  if (id_ == 0) {
    result = &ano_;
  } else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.first;
  }
  return *result;
}

const std::string& DDName::ns() const {
  const static std::string ano_("anonymous");
  const std::string* result;
  if (id_ == 0) {
    result = &ano_;
  } else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.second;
  }
  return *result;
}

namespace {
  std::once_flag s_once;
}  // namespace

DDName::Registry::const_iterator DDName::registerName(const std::pair<std::string, std::string>& nm) {
  std::call_once(s_once, []() {
    Registry& reg = DDI::Singleton<Registry>::instance();
    IdToName& idToName = DDI::Singleton<IdToName>::instance();
    reg.emplace(std::make_pair(std::string(""), std::string("")), 0);
    idToName.emplace_back(reg.begin());
  });
  Registry& reg = DDI::Singleton<Registry>::instance();

  Registry::const_iterator itFound = reg.find(nm);
  if (itFound == reg.end()) {
    //If two threads are concurrently adding the same name we will get
    // two entries in IdToName but they will both point to the same entry
    // to Registry where the first emplace to Registry will set the ID number.
    IdToName& idToName = DDI::Singleton<IdToName>::instance();
    auto it = idToName.emplace_back(reg.end());
    *it = reg.emplace(nm, it - idToName.begin()).first;
    itFound = *it;
  }
  return itFound;
}
