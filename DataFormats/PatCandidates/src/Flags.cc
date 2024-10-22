#include "DataFormats/PatCandidates/interface/Flags.h"

using pat::Flags;

const std::string &Flags::bitToString(uint32_t bit) {
  static const std::string UNDEFINED_UNDEFINED = "Undefined/Undefined";
  if (bit & CoreBits)
    return Core::bitToString(Core::Bits(bit));
  else if (bit & SelectionBits)
    return Selection::bitToString(Selection::Bits(bit));
  else if (bit & OverlapBits)
    return Overlap::bitToString(Overlap::Bits(bit));
  else if (bit & IsolationBits)
    return Isolation::bitToString(Isolation::Bits(bit));
  else
    return UNDEFINED_UNDEFINED;
}

std::string Flags::maskToString(uint32_t mask) {
  std::string ret;
  bool first = false;
  for (uint32_t i = 1; i != 0; i <<= 1) {
    if (mask & i) {
      if (first) {
        first = false;
      } else {
        ret += " + ";
      }
      ret += bitToString(mask & i);
    }
  }
  return ret;
}

uint32_t Flags::get(const std::string &str) {
  size_t idx = str.find_first_of('/');
  if (idx != std::string::npos) {
    std::string set = str.substr(0, idx);
    if (set == "Core")
      return Core::get(str.substr(idx + 1));
    if (set == "Overlap")
      return Overlap::get(str.substr(idx + 1));
    if (set == "Isolation")
      return Isolation::get(str.substr(idx + 1));
    if (set == "Selection")
      return Selection::get(str.substr(idx + 1));
  }
  return 0;
}

uint32_t Flags::get(const std::vector<std::string> &strs) {
  uint32_t ret = 0;
  for (std::vector<std::string>::const_iterator it = strs.begin(), ed = strs.end(); it != ed; ++it) {
    ret |= get(*it);
  }
  return ret;
}

const std::string &Flags::Core::bitToString(Core::Bits bit) {
  static const std::string STR_All = "Core/All", STR_Duplicate = "Core/Duplicate",
                           STR_Preselection = "Core/Preselection", STR_Vertexing = "Core/Vertexing",
                           STR_Overflow = "Core/Overflow", STR_Undefined = "Core/Undefined";

  switch (bit) {
    case All:
      return STR_All;
    case Duplicate:
      return STR_Duplicate;
    case Preselection:
      return STR_Preselection;
    case Vertexing:
      return STR_Vertexing;
    case Overflow:
      return STR_Overflow;
    default:
      return STR_Undefined;
  }
}

Flags::Core::Bits Flags::Core::get(const std::string &instr) {
  size_t idx = instr.find_first_of('/');
  const std::string &str = (idx == std::string::npos) ? instr : instr.substr(idx + 1);
  if (str == "All")
    return All;
  else if (str == "Duplicate")
    return Duplicate;
  else if (str == "Preselection")
    return Preselection;
  else if (str == "Vertexing")
    return Vertexing;
  else if (str == "Overlfow")
    return Overflow;
  return Undefined;
}

uint32_t Flags::Core::get(const std::vector<std::string> &strs) {
  uint32_t ret = 0;
  for (std::vector<std::string>::const_iterator it = strs.begin(), ed = strs.end(); it != ed; ++it) {
    ret |= get(*it);
  }
  return ret;
}

const std::string &Flags::Selection::bitToString(Selection::Bits bit) {
  static const std::string STR_All = "Selection/All", STR_Bit0 = "Selection/Bit0", STR_Bit1 = "Selection/Bit1",
                           STR_Bit2 = "Selection/Bit2", STR_Bit3 = "Selection/Bit3", STR_Bit4 = "Selection/Bit4",
                           STR_Bit5 = "Selection/Bit5", STR_Bit6 = "Selection/Bit6", STR_Bit7 = "Selection/Bit7",
                           STR_Bit8 = "Selection/Bit8", STR_Bit9 = "Selection/Bit9", STR_Bit10 = "Selection/Bit10",
                           STR_Bit11 = "Selection/Bit11", STR_Undefined = "Selection/Undefined";
  switch (bit) {
    case All:
      return STR_All;
    case Bit0:
      return STR_Bit0;
    case Bit1:
      return STR_Bit1;
    case Bit2:
      return STR_Bit2;
    case Bit3:
      return STR_Bit3;
    case Bit4:
      return STR_Bit4;
    case Bit5:
      return STR_Bit5;
    case Bit6:
      return STR_Bit6;
    case Bit7:
      return STR_Bit7;
    case Bit8:
      return STR_Bit8;
    case Bit9:
      return STR_Bit9;
    case Bit10:
      return STR_Bit10;
    case Bit11:
      return STR_Bit11;
    default:
      return STR_Undefined;
  }
}

Flags::Selection::Bits Flags::Selection::get(int8_t bit) {
  if (bit == -1)
    return All;
  if (bit <= 11)
    return Bits((1 << bit) << 8);
  return Undefined;
}

Flags::Selection::Bits Flags::Selection::get(const std::string &instr) {
  size_t idx = instr.find_first_of('/');
  const std::string &str = (idx == std::string::npos) ? instr : instr.substr(idx + 1);
  if (str == "All")
    return All;
  else if (str == "Bit0")
    return Bit0;
  else if (str == "Bit1")
    return Bit1;
  else if (str == "Bit2")
    return Bit2;
  else if (str == "Bit3")
    return Bit3;
  else if (str == "Bit4")
    return Bit4;
  else if (str == "Bit5")
    return Bit5;
  else if (str == "Bit6")
    return Bit6;
  else if (str == "Bit7")
    return Bit7;
  else if (str == "Bit8")
    return Bit8;
  else if (str == "Bit9")
    return Bit9;
  else if (str == "Bit10")
    return Bit10;
  else if (str == "Bit11")
    return Bit11;
  return Undefined;
}

uint32_t Flags::Selection::get(const std::vector<std::string> &strs) {
  uint32_t ret = 0;
  for (std::vector<std::string>::const_iterator it = strs.begin(), ed = strs.end(); it != ed; ++it) {
    ret |= get(*it);
  }
  return ret;
}

const std::string &Flags::Overlap::bitToString(Overlap::Bits bit) {
  static const std::string STR_All = "Overlap/All", STR_Jets = "Overlap/Jets", STR_Electrons = "Overlap/Electrons",
                           STR_Muons = "Overlap/Muons", STR_Taus = "Overlap/Taus", STR_Photons = "Overlap/Photons",
                           STR_User = "Overlap/User", STR_User1 = "Overlap/User1", STR_User2 = "Overlap/User2",
                           STR_User3 = "Overlap/User3", STR_Undefined = "Overlap/Undefined";
  switch (bit) {
    case All:
      return STR_All;
    case Jets:
      return STR_Jets;
    case Electrons:
      return STR_Electrons;
    case Muons:
      return STR_Muons;
    case Taus:
      return STR_Taus;
    case Photons:
      return STR_Photons;
    case User:
      return STR_User;
    case User1:
      return STR_User1;
    case User2:
      return STR_User2;
    case User3:
      return STR_User3;
    default:
      return STR_Undefined;
  }
}

Flags::Overlap::Bits Flags::Overlap::get(const std::string &instr) {
  size_t idx = instr.find_first_of('/');
  const std::string &str = (idx == std::string::npos) ? instr : instr.substr(idx + 1);
  if (str == "All")
    return All;
  else if (str == "Jets")
    return Jets;
  else if (str == "Electrons")
    return Electrons;
  else if (str == "Muons")
    return Muons;
  else if (str == "Taus")
    return Taus;
  else if (str == "Photons")
    return Photons;
  else if (str == "User")
    return User;
  else if (str == "User1")
    return User1;
  else if (str == "User2")
    return User2;
  else if (str == "User3")
    return User3;
  return Undefined;
}

uint32_t Flags::Overlap::get(const std::vector<std::string> &strs) {
  uint32_t ret = 0;
  for (std::vector<std::string>::const_iterator it = strs.begin(), ed = strs.end(); it != ed; ++it) {
    ret |= get(*it);
  }
  return ret;
}

const std::string &Flags::Isolation::bitToString(Isolation::Bits bit) {
  static const std::string STR_All = "Isolation/All", STR_Tracker = "Isolation/Tracker", STR_ECal = "Isolation/ECal",
                           STR_HCal = "Isolation/HCal", STR_Calo = "Isolation/Calo", STR_User = "Isolation/User",
                           STR_User1 = "Isolation/User1", STR_User2 = "Isolation/User2", STR_User3 = "Isolation/User3",
                           STR_User4 = "Isolation/User4", STR_User5 = "Isolation/User5",
                           STR_Undefined = "Isolation/Undefined";
  switch (bit) {
    case All:
      return STR_All;
    case Tracker:
      return STR_Tracker;
    case ECal:
      return STR_ECal;
    case HCal:
      return STR_HCal;
    case Calo:
      return STR_Calo;
    case User:
      return STR_User;
    case User1:
      return STR_User1;
    case User2:
      return STR_User2;
    case User3:
      return STR_User3;
    case User4:
      return STR_User4;
    case User5:
      return STR_User5;
    default:
      return STR_Undefined;
  }
}

Flags::Isolation::Bits Flags::Isolation::get(const std::string &instr) {
  size_t idx = instr.find_first_of('/');
  const std::string &str = (idx == std::string::npos) ? instr : instr.substr(idx + 1);
  if (str == "All")
    return All;
  else if (str == "Tracker")
    return Tracker;
  else if (str == "ECal")
    return ECal;
  else if (str == "HCal")
    return HCal;
  else if (str == "Calo")
    return Calo;
  else if (str == "User")
    return User1;
  else if (str == "User1")
    return User1;
  else if (str == "User2")
    return User2;
  else if (str == "User3")
    return User3;
  else if (str == "User4")
    return User4;
  else if (str == "User5")
    return User5;
  return Undefined;
}

uint32_t Flags::Isolation::get(const std::vector<std::string> &strs) {
  uint32_t ret = 0;
  for (std::vector<std::string>::const_iterator it = strs.begin(), ed = strs.end(); it != ed; ++it) {
    ret |= get(*it);
  }
  return ret;
}
