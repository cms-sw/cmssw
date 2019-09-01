#include "DQMOffline/Trigger/interface/EgHLTComCodes.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace egHLT;

void ComCodes::setCode(const char* descript, int code) {
  bool found = false;
  for (size_t i = 0; i < _codeDefs.size() && !found; i++) {
    if (_codeDefs[i].first == descript)
      found = true;
  }
  if (!found)
    _codeDefs.emplace_back(descript, code);

  //_codeDefs[descript] = code;
}

int ComCodes::getCode(const char* descript) const {
  int code = 0x0000;
  char const* const end = descript + strlen(descript);
  char const* codeKey = descript;
  char const* token = nullptr;
  do {
    token = std::find(codeKey, end, ':');

    bool found = false;
    for (auto const& c : _codeDefs) {
      if (0 == c.first.compare(0, std::string::npos, codeKey, token - codeKey)) {
        code |= c.second;
        found = true;
        break;
      }
    }
    if (!found)
      edm::LogWarning("EgHLTComCodes")
          << "ComCodes::getCode : Error, Key " << std::string(codeKey, token - codeKey)
          << " not found (likely mistyped, practical upshot is the selection is not what you think it is)";  //<<std::endl;
    codeKey = token + 1;
  } while (token != end);
  return code;
}

bool ComCodes::keyComp(const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs) {
  return lhs.first < rhs.first;
}

void ComCodes::getCodeName(int code, std::string& id) const {
  id.clear();
  for (auto const& _codeDef : _codeDefs) {
    if ((code & _codeDef.second) == _codeDef.second) {
      if (!id.empty())
        id += ":";  //seperating entries by a ':'
      id += _codeDef.first;
    }
  }
}
