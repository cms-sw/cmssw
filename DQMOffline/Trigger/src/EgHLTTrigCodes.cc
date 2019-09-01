#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

const int egHLT::TrigCodes::maxNrBits_;

using namespace egHLT;

TrigCodes* TrigCodes::makeCodes(std::vector<std::string>& filterNames) {
  auto* p = new TrigCodes();

  for (size_t i = 0; i < filterNames.size(); i++) {
    p->setCode(filterNames[i].c_str(), i);
  }
  p->sort();

  return p;
}

void TrigCodes::setCode(const char* descript, int bitNr) {
  if (bitNr < maxNrBits_) {
    TrigBitSet code;
    code.set(bitNr);
    setCode(descript, code);
  } else {
    edm::LogWarning("TrigCodes::TrigBitSetMap")
        << " Warning, trying to store at bit " << bitNr << " but max nr bits is " << maxNrBits_;
  }
}

void TrigCodes::setCode(const char* descript, TrigBitSet code) {
  bool found = false;
  for (size_t i = 0; i < codeDefs_.size() && !found; i++) {
    if (codeDefs_[i].first == descript)
      found = true;
  }
  if (!found)
    codeDefs_.emplace_back(descript, code);
  //_codeDefs[descript] = code;
}

TrigCodes::TrigBitSet TrigCodes::getCode(const char* descript) const {
  TrigBitSet code;

  char const* const end = descript + strlen(descript);
  char const* codeKey = descript;
  char const* token = nullptr;
  do {
    token = std::find(codeKey, end, ':');

    for (auto const& c : codeDefs_) {
      if (0 == c.first.compare(0, std::string::npos, codeKey, token - codeKey)) {
        code |= c.second;
        break;
      }
    }
    codeKey = token + 1;
  } while (token != end);
  return code;
}

bool TrigCodes::keyComp(const std::pair<std::string, TrigBitSet>& lhs, const std::pair<std::string, TrigBitSet>& rhs) {
  return lhs.first < rhs.first;
}

void TrigCodes::getCodeName(TrigBitSet code, std::string& id) const {
  id.clear();
  for (auto const& codeDef : codeDefs_) {
    if ((code & codeDef.second) == codeDef.second) {
      if (!id.empty())
        id += ":";  //seperating entries by a ':'
      id += codeDef.first;
    }
  }
}

void TrigCodes::printCodes() {
  std::ostringstream msg;
  msg << " trig bits defined: " << std::endl;
  for (auto& codeDef : codeDefs_)
    msg << " key : " << codeDef.first << " bit " << codeDef.second << std::endl;
  edm::LogInfo("TrigCodes") << msg.str();
}
