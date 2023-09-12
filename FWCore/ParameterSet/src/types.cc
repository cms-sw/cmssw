// ----------------------------------------------------------------------
// definition of type encoding/decoding functions
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/types.h"

#include "FWCore/ParameterSet/src/split.h"
#include "FWCore/Utilities/interface/Parse.h"
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <optional>

using namespace edm;

// ----------------------------------------------------------------------
// utility functions
// ----------------------------------------------------------------------

static char to_hex(unsigned int i) { return i + (i < 10u ? '0' : ('A' - 10)); }

// ----------------------------------------------------------------------

static unsigned int from_hex(char c) {
  switch (c) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return c - '0';
    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'e':
    case 'f':
      return 10 + c - 'a';
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
      return 10 + c - 'A';
    default:
      return 0;
  }
}  // from_hex()

static void append_hex_rep(std::string& s, unsigned int c) {
  s += to_hex(c / 16u);
  s += to_hex(c % 16u);
}  // append_hex_rep()

// ----------------------------------------------------------------------
// Bool
// ----------------------------------------------------------------------

bool edm::decode(bool& to, std::string_view from) {
  if (from == "true") {
    to = true;
    return true;
  } else if (from == "false") {
    to = false;
    return true;
  } else {
    return false;
  }
}  // decode to bool

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, bool from) {
  to = from ? "true" : "false";
  return true;
}  // encode from bool

// ----------------------------------------------------------------------
// vBool
// ----------------------------------------------------------------------

bool edm::decode(std::vector<bool>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    bool val = false;
    if (!decode(val, t)) {
      return false;
    }
    to.push_back(val);
    return true;
  });
}  // decode to vector<bool>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<bool> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<bool>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }
    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }
  to += '}';
  return true;
}  // encode from vector<bool>

// ----------------------------------------------------------------------
// Int32
// ----------------------------------------------------------------------

bool edm::decode(int& to, std::string_view from) {
  std::string_view::const_iterator b = from.begin(), e = from.end();

  if (*b != '+' && *b != '-') {
    return false;
  }
  int sign = (*b == '+') ? +1 : -1;

  to = 0;
  while (++b != e) {
    if (!std::isdigit(*b)) {
      return false;
    }
    to = 10 * to + (*b - '0');
  }
  to *= sign;

  return true;
}  // decode to int

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, int from) {
  bool is_negative = (from < 0);
  if (is_negative) {
    from = -from;  // TODO: work around this for most negative integer
  }
  to.clear();
  do {
    to = static_cast<char>(from % 10 + '0') + to;
    from /= 10;
  } while (from > 0);
  to = (is_negative ? '-' : '+') + to;

  return true;
}  // encode from int

// ----------------------------------------------------------------------
// Int64
// ----------------------------------------------------------------------

bool edm::decode(long long& to, std::string_view from) {
  std::string_view::const_iterator b = from.begin(), e = from.end();

  if (*b != '+' && *b != '-') {
    return false;
  }
  int sign = (*b == '+') ? +1 : -1;

  to = 0;
  while (++b != e) {
    if (!std::isdigit(*b)) {
      return false;
    }
    to = 10 * to + (*b - '0');
  }
  to *= sign;

  return true;
}  // decode to int

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, long long from) {
  bool is_negative = (from < 0);
  if (is_negative) {
    from = -from;  // TODO: work around this for most negative integer
  }

  to.clear();
  do {
    to = static_cast<char>(from % 10 + '0') + to;
    from /= 10;
  } while (from > 0);
  to = (is_negative ? '-' : '+') + to;

  return true;
}  // encode from int

// ----------------------------------------------------------------------
// vInt32
// ----------------------------------------------------------------------

bool edm::decode(std::vector<int>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    int val = 0;
    if (!decode(val, t)) {
      return false;
    }
    to.push_back(val);
    return true;
  });
}  // decode to vector<int>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<int> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<int>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }

    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<int>

// ----------------------------------------------------------------------
// vInt64
// ----------------------------------------------------------------------

bool edm::decode(std::vector<long long>& to, std::string_view from) {
  std::vector<std::string_view> temp;
  if (!split(std::back_inserter(temp), from, '{', ',', '}')) {
    return false;
  }

  to.clear();
  for (auto t : temp) {
    long long val = 0LL;
    if (!decode(val, t)) {
      return false;
    }
    to.push_back(val);
  }

  return true;
}  // decode to vector<int>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<long long> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<long long>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }
    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }
  to += '}';
  return true;
}  // encode from vector<int>

// ----------------------------------------------------------------------
// Uint32
// ----------------------------------------------------------------------

bool edm::decode(unsigned int& to, std::string_view from) {
  std::string_view::const_iterator b = from.begin(), e = from.end();

  to = 0u;
  for (; b != e; ++b) {
    if (*b == 'u' || *b == 'U') {
      return true;
    }
    if (!std::isdigit(*b)) {
      return false;
    }
    to = 10u * to + (*b - '0');
  }
  return true;
}  // decode to unsigned

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, unsigned int from) {
  to.clear();
  do {
    to = static_cast<char>(from % 10 + '0') + to;
    from /= 10u;
  } while (from > 0u);

  return true;
}  // encode from unsigned

// ----------------------------------------------------------------------
// Uint64
// ----------------------------------------------------------------------

bool edm::decode(unsigned long long& to, std::string_view from) {
  std::string_view::const_iterator b = from.begin(), e = from.end();
  to = 0u;
  for (; b != e; ++b) {
    if (*b == 'u' || *b == 'U') {
      return true;
    }
    if (!std::isdigit(*b)) {
      return false;
    }
    to = 10u * to + (*b - '0');
  }
  return true;
}  // decode to unsigned

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, unsigned long long from) {
  to.clear();
  do {
    to = static_cast<char>(from % 10 + '0') + to;
    from /= 10u;
  } while (from > 0u);

  return true;
}  // encode from unsigned

// ----------------------------------------------------------------------
// vUint32
// ----------------------------------------------------------------------

bool edm::decode(std::vector<unsigned int>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    unsigned int val = 0;
    if (!decode(val, t)) {
      return false;
    }
    to.push_back(val);
    return true;
  });
}  // decode to vector<unsigned int>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<unsigned int> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<unsigned int>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }
    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<unsigned int>

// ----------------------------------------------------------------------
// vUint64
// ----------------------------------------------------------------------

bool edm::decode(std::vector<unsigned long long>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    unsigned long long val = 0ULL;
    if (!decode(val, t)) {
      return false;
    }
    to.push_back(val);
    return true;
  });
}  // decode to vector<unsigned int>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<unsigned long long> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<unsigned long long>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }

    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<unsigned int>

// ----------------------------------------------------------------------
// Double
// ----------------------------------------------------------------------

bool edm::decode(double& to, std::string_view from) {
  if (from == "NaN") {
    to = std::numeric_limits<double>::quiet_NaN();
  } else if (from == "+inf" || from == "inf") {
    to = std::numeric_limits<double>::has_infinity ? std::numeric_limits<double>::infinity()
                                                   : std::numeric_limits<double>::max();
  } else if (from == "-inf") {
    to = std::numeric_limits<double>::has_infinity ? -std::numeric_limits<double>::infinity()
                                                   : -std::numeric_limits<double>::max();
  }

  else {
    try {
      // std::cerr << "from:" << from << std::endl;
      to = std::stod(std::string(from));
      // std::cerr << "to:" << to << std::endl;
    } catch (const std::exception&) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, double from) {
  std::ostringstream ost;
  ost.precision(std::numeric_limits<double>::max_digits10);
  ost << from;
  if (!ost)
    return false;
  to = ost.str();
  return true;
}

// ----------------------------------------------------------------------
// vDouble
// ----------------------------------------------------------------------

bool edm::decode(std::vector<double>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    double val;
    if (!decode(val, t))
      return false;
    to.push_back(val);
    return true;
  });
}  // decode to vector<double>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<double> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<double>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b))
      return false;

    if (b != from.begin())
      to += ",";
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<double>

// ----------------------------------------------------------------------
// String
// ----------------------------------------------------------------------
std::optional<std::string_view> edm::decode_string_extent(std::string_view from) {
  std::size_t searchIndex = 0;
  std::size_t indexEnd = 0;
  while (std::string_view::npos != (indexEnd = from.find_first_of('\0', searchIndex))) {
    if (indexEnd + 1 == from.size()) {
      return from;
    }
    if (from[indexEnd + 1] == '\0') {
      searchIndex = indexEnd + 2;
    } else {
      return from.substr(0, indexEnd + 1);
    }
  }
  //didn't find an unpaired '\0'
  return {};
}

bool edm::decode(std::string& to, std::string_view from) {
  if (from.empty() or from.back() != '\0') {
    return false;
  }
  to = from.substr(0, from.size() - 1);

  std::size_t searchIndex = 0;
  while (std::string::npos != (searchIndex = to.find_first_of('\0', searchIndex))) {
    if (searchIndex == to.size() - 1 or to[searchIndex + 1] != '\0') {
      //unpaired string
      return false;
    }
    to.erase(searchIndex, 1);
    ++searchIndex;
  }
  return true;
}

bool edm::decode_deprecated(std::string& to, std::string_view from) {
  /*std::cerr << "Decoding: " << from << '\n'; //DEBUG*/
  std::string_view::const_iterator b = from.begin(), e = from.end();

  to = "";
  to.reserve((e - b) / 2);
  char c = '\0';
  for (bool even_pos = true; b != e; ++b, even_pos = !even_pos) {
    if (even_pos) {
      /*std::cerr << "Even: |"
                  << *b
                  << "|   giving "
                  << from_hex(*b)
                  << "\n"; //DEBUG*/
      c = static_cast<char>(from_hex(*b));
    } else {
      /*std::cerr << "Odd:  |"
                  << *b
                  << "|   giving "
                  << from_hex(*b)
                  << "\n"; //DEBUG*/
      c = static_cast<char>(c * 16 + from_hex(*b));
      //      if(std::isalnum(c))  {
      /*std::cerr << "Ans:  |" << c << "|\n"; //DEBUG*/
      to += c;
      //}
      //else  { // keep all special chars encoded
      //to += "\\x";
      //to += to_hex_rep(c);
      //}
    }
  }
  /*std::cerr << "Decoded: " << to << '\n'; //DEBUG*/
  return true;
}  // decode to String

// ----------------------------------------------------------------------
// FileInPath
// ----------------------------------------------------------------------

bool edm::decode(FileInPath& to, std::string_view from) {
  std::string sfrom{from};
  std::istringstream is(sfrom);
  FileInPath temp;
  temp.readFromParameterSetBlob(is);
  if (!is)
    return false;
  to = temp;
  return true;
}  // decode to FileInPath

bool edm::encode(std::string& to, FileInPath const& from) {
  std::ostringstream ost;
  ost << from;
  if (!ost)
    return false;
  to = ost.str();
  return true;
}

// ----------------------------------------------------------------------
// InputTag
// ----------------------------------------------------------------------

bool edm::decode(InputTag& to, std::string_view from) {
  to = InputTag(std::string(from));
  return true;
}  // decode to InputTag

bool edm::encode(std::string& to, InputTag const& from) {
  to = from.encode();
  return true;
}

// ----------------------------------------------------------------------
// VInputTag
// ----------------------------------------------------------------------

bool edm::decode(std::vector<InputTag>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    to.push_back(InputTag(*stringItr));
  }
  return true;
}  // decode to VInputTag

bool edm::encode(std::string& to, std::vector<InputTag> const& from) {
  std::vector<std::string> strings;
  for (std::vector<InputTag>::const_iterator tagItr = from.begin(), tagItrEnd = from.end(); tagItr != tagItrEnd;
       ++tagItr) {
    strings.push_back(tagItr->encode());
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
// ESInputTag
// ----------------------------------------------------------------------

bool edm::decode(ESInputTag& to, std::string_view from) {
  if (not from.empty() and from.npos == from.find(':')) {
    to = ESInputTag(std::string(from), "");
  } else {
    to = ESInputTag(std::string(from));
  }
  return true;
}  // decode to InputTag

bool edm::encode(std::string& to, ESInputTag const& from) {
  to = from.encode();
  if (not to.empty() and to.back() == ':') {
    to.pop_back();
  }
  return true;
}

// ----------------------------------------------------------------------
// VESInputTag
// ----------------------------------------------------------------------

bool edm::decode(std::vector<ESInputTag>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    to.push_back(ESInputTag(*stringItr));
  }
  return true;
}  // decode to VInputTag

bool edm::encode(std::string& to, std::vector<ESInputTag> const& from) {
  std::vector<std::string> strings;
  for (std::vector<ESInputTag>::const_iterator tagItr = from.begin(), tagItrEnd = from.end(); tagItr != tagItrEnd;
       ++tagItr) {
    strings.push_back(tagItr->encode());
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
// EventID
// ----------------------------------------------------------------------

bool edm::decode(edm::EventID& to, std::string_view from) {
  std::vector<std::string> tokens = edm::tokenize(std::string(from), ":");
  assert(tokens.size() == 2 || tokens.size() == 3);
  unsigned int run = strtoul(tokens[0].c_str(), nullptr, 0);
  unsigned int lumi = (tokens.size() == 2 ? 0 : strtoul(tokens[1].c_str(), nullptr, 0));
  unsigned long long event = strtoull(tokens[tokens.size() - 1].c_str(), nullptr, 0);
  to = edm::EventID(run, lumi, event);

  return true;
}  // decode to EventID

bool edm::encode(std::string& to, edm::EventID const& from) {
  std::ostringstream os;
  if (from.luminosityBlock() == 0U) {
    os << from.run() << ":" << from.event();
  } else {
    os << from.run() << ":" << from.luminosityBlock() << ":" << from.event();
  }
  to = os.str();
  return true;
}

// ----------------------------------------------------------------------
// VEventID
// ----------------------------------------------------------------------

bool edm::decode(std::vector<edm::EventID>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    edm::EventID eventID;
    decode(eventID, *stringItr);
    to.push_back(eventID);
  }
  return true;
}  // decode to VInputTag

bool edm::encode(std::string& to, std::vector<edm::EventID> const& from) {
  std::vector<std::string> strings;
  for (std::vector<edm::EventID>::const_iterator idItr = from.begin(), idItrEnd = from.end(); idItr != idItrEnd;
       ++idItr) {
    std::string encodedEventID;
    encode(encodedEventID, *idItr);
    strings.push_back(encodedEventID);
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
// LuminosityBlockID
// ----------------------------------------------------------------------

bool edm::decode(edm::LuminosityBlockID& to, std::string_view from) {
  std::vector<std::string> tokens = edm::tokenize(std::string(from), ":");
  assert(tokens.size() == 2);
  unsigned int run = strtoul(tokens[0].c_str(), nullptr, 0);
  unsigned int lumi = strtoul(tokens[1].c_str(), nullptr, 0);
  to = edm::LuminosityBlockID(run, lumi);
  return true;
}  // decode to LuminosityBlockID

bool edm::encode(std::string& to, edm::LuminosityBlockID const& from) {
  std::ostringstream os;
  os << from.run() << ":" << from.luminosityBlock();
  to = os.str();
  return true;
}

// ----------------------------------------------------------------------
// VLuminosityBlockID
// ----------------------------------------------------------------------

bool edm::decode(std::vector<edm::LuminosityBlockID>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    edm::LuminosityBlockID lumiID;
    decode(lumiID, *stringItr);
    to.push_back(lumiID);
  }
  return true;
}  // decode to VInputTag

bool edm::encode(std::string& to, std::vector<edm::LuminosityBlockID> const& from) {
  std::vector<std::string> strings;
  for (std::vector<edm::LuminosityBlockID>::const_iterator idItr = from.begin(), idItrEnd = from.end();
       idItr != idItrEnd;
       ++idItr) {
    std::string encodedLuminosityBlockID;
    encode(encodedLuminosityBlockID, *idItr);
    strings.push_back(encodedLuminosityBlockID);
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
// LuminosityBlockRange
// ----------------------------------------------------------------------

bool edm::decode(edm::LuminosityBlockRange& to, std::string_view from) {
  std::vector<std::string> tokens = edm::tokenize(std::string(from), "-");
  assert(tokens.size() == 2);
  edm::LuminosityBlockID begin;
  edm::LuminosityBlockID end;
  edm::decode(begin, tokens[0]);
  edm::decode(end, tokens[1]);
  to = edm::LuminosityBlockRange(begin.run(), begin.luminosityBlock(), end.run(), end.luminosityBlock());
  return true;
}  // decode to LuminosityBlockRange

bool edm::encode(std::string& to, edm::LuminosityBlockRange const& from) {
  std::ostringstream os;
  os << from.startRun() << ":" << from.startLumi() << "-" << from.endRun() << ":" << from.endLumi();
  to = os.str();
  return true;
}

// ----------------------------------------------------------------------
// VLuminosityBlockRange
// ----------------------------------------------------------------------

bool edm::decode(std::vector<edm::LuminosityBlockRange>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    edm::LuminosityBlockRange lumiRange;
    decode(lumiRange, *stringItr);
    to.push_back(lumiRange);
  }
  return true;
}  // decode to VInputTag

bool edm::encode(std::string& to, std::vector<edm::LuminosityBlockRange> const& from) {
  std::vector<std::string> strings;
  for (std::vector<edm::LuminosityBlockRange>::const_iterator idItr = from.begin(), idItrEnd = from.end();
       idItr != idItrEnd;
       ++idItr) {
    std::string encodedLuminosityBlockRange;
    encode(encodedLuminosityBlockRange, *idItr);
    strings.push_back(encodedLuminosityBlockRange);
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
// EventRange
// ----------------------------------------------------------------------

bool edm::decode(edm::EventRange& to, std::string_view from) {
  std::vector<std::string> tokens = edm::tokenize(std::string(from), "-");
  assert(tokens.size() == 2);
  edm::EventID begin;
  edm::EventID end;
  edm::decode(begin, tokens[0]);
  edm::decode(end, tokens[1]);
  assert((begin.luminosityBlock() == 0) == (end.luminosityBlock() == 0));
  to = edm::EventRange(
      begin.run(), begin.luminosityBlock(), begin.event(), end.run(), end.luminosityBlock(), end.event());
  return true;
}  // decode to EventRange

bool edm::encode(std::string& to, edm::EventRange const& from) {
  std::ostringstream os;
  if (from.startLumi() == 0) {
    assert(from.endLumi() == 0);
    os << from.startRun() << ":" << from.startEvent() << "-" << from.endRun() << ":" << from.endEvent();
  } else {
    assert(from.endLumi() != 0);
    os << from.startRun() << ":" << from.startLumi() << ":" << from.startEvent() << "-" << from.endRun() << ":"
       << from.endLumi() << ":" << from.endEvent();
  }
  to = os.str();
  return true;
}

// ----------------------------------------------------------------------
// VEventRange
// ----------------------------------------------------------------------

bool edm::decode(std::vector<edm::EventRange>& to, std::string_view from) {
  std::vector<std::string> strings;
  decode(strings, from);

  for (std::vector<std::string>::const_iterator stringItr = strings.begin(), stringItrEnd = strings.end();
       stringItr != stringItrEnd;
       ++stringItr) {
    edm::EventRange eventRange;
    decode(eventRange, *stringItr);
    to.push_back(eventRange);
  }
  return true;
}

bool edm::encode(std::string& to, std::vector<edm::EventRange> const& from) {
  std::vector<std::string> strings;
  for (std::vector<edm::EventRange>::const_iterator idItr = from.begin(), idItrEnd = from.end(); idItr != idItrEnd;
       ++idItr) {
    std::string encodedEventRange;
    encode(encodedEventRange, *idItr);
    strings.push_back(encodedEventRange);
  }
  encode(to, strings);
  return true;
}

// ----------------------------------------------------------------------
bool edm::encode(std::string& to, std::string const& from) {
  to = from;
  //need to escape any nulls by making them pairs
  std::size_t lastFound = 0;
  while (std::string::npos != (lastFound = to.find_first_of('\0', lastFound))) {
    to.insert(lastFound, 1, '\0');
    lastFound += 2;
  }
  to += '\0';
  return true;
}

bool edm::encode_deprecated(std::string& to, std::string const& from) {
  std::string::const_iterator b = from.begin(), e = from.end();

  enum escape_state { NONE, BACKSLASH, HEX, HEX1, OCT1, OCT2 };

  escape_state state = NONE;
  int code = 0;
  to = "";
  for (; b != e; ++b) {
    /*std::cerr << "State: " << state << "; char = " << *b << '\n'; //DEBUG*/
    switch (state) {
      case NONE: {
        if (*b == '\\')
          state = BACKSLASH;
        else
          append_hex_rep(to, *b);
        /*std::cerr << "To: |" << to << "|\n"; //DEBUG*/
        break;
      }
      case BACKSLASH: {
        code = 0;
        switch (*b) {
          case 'x':
          case 'X': {
            state = HEX;
            break;
          }
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7': {
            code = 8 * code + from_hex(*b);
            state = OCT1;
            break;
          }
          case 'n': {
            append_hex_rep(to, 10);
            state = NONE;
            break;
          }
          case 't': {
            append_hex_rep(to, 9);
            state = NONE;
            break;
          }
          default: {
            append_hex_rep(to, *b);
            state = NONE;
            break;
          }
        }
        break;
      }
      case HEX: {
        to += *b;
        state = HEX1;
        break;
      }
      case HEX1: {
        to += *b;
        state = NONE;
        break;
      }
      case OCT1: {
        switch (*b) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7': {
            code = 8 * code + from_hex(*b);
            state = OCT2;
            break;
          }
          default: {
            append_hex_rep(to, code);
            state = NONE;
            break;
          }
        }
        break;
      }
      case OCT2: {
        switch (*b) {
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7': {
            code = 8 * code + from_hex(*b);
            break;
          }
          default: {
            append_hex_rep(to, code);
            break;
          }
        }
        state = NONE;
        break;
      }
      default: {
        throw std::logic_error("can't happen");
        break;
      }
    }
  }  // for

  return true;
}  // encode from String

// ----------------------------------------------------------------------
// vString
// ----------------------------------------------------------------------

bool edm::decode_deprecated(std::vector<std::string>& to, std::string_view from) {
  to.clear();
  to.reserve(std::count(from.begin(), from.end(), ','));
  return split(from, '{', ',', '}', [&to](auto t) {
    std::string val;
    // treat blank string specially
    if (t == "XXX") {
      val = "";
    } else if (!decode_deprecated(val, t)) {
      return false;
    }
    to.push_back(val);
    return true;
  });
}  // decode to vector<string>

std::optional<std::string_view> edm::decode_vstring_extent(std::string_view from) {
  if (from.size() < 2) {
    return {};
  }
  if (from.front() != '{') {
    return {};
  }
  if (from[1] == '}') {
    return from.substr(0, 2);
  }
  if (from.size() < 3) {
    return {};
  }
  if (from[1] != '\0') {
    return {};
  }
  auto remaining = from.substr(2);
  while (not remaining.empty()) {
    auto strng = decode_string_extent(remaining);
    if (not strng) {
      return {};
    }
    remaining = remaining.substr(strng->size());
    if (remaining.empty())
      return {};
    if (remaining.front() == '}') {
      return from.substr(0, from.size() - remaining.size() + 1);
    }
    if (remaining.front() == ',') {
      remaining = remaining.substr(1);
    } else {
      return {};
    }
  }
  return {};
}

bool edm::decode(std::vector<std::string>& to, std::string_view from) {
  if (from.size() < 2) {
    return false;
  }
  if (from.front() != '{') {
    return false;
  }
  if (from.back() != '}') {
    return false;
  }
  to.clear();
  if (from.size() == 2) {
    //an empty vector
    return true;
  }
  if (from[1] != '\0') {
    return false;
  }
  auto remaining = from.substr(2, from.size() - 2 /*leading {/0*/ - 1 /*trailing } */);
  while (not remaining.empty()) {
    auto strng = decode_string_extent(remaining);
    if (not strng) {
      return false;
    }
    std::string val;
    if (!decode_element(val, *strng)) {
      return false;
    }
    to.emplace_back(std::move(val));

    remaining = remaining.substr(strng->size());
    if (remaining.empty())
      break;
    if (remaining.front() == ',') {
      remaining = remaining.substr(1);
    } else {
      return false;
    }
  }
  return true;
}  // decode to vector<string>

bool edm::decode_element(std::string& to, std::string_view from) { return decode(to, from); }

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<std::string> const& from) {
  to = "{";

  //special cases
  // empty vector : "{}"
  // vector with one empty element: "{\0\0}"
  // vector with one '}' element: "{\0}\0}"
  // vector with two '}' elements: "{\0}\0,}\0}"
  //needed to tell an empty vector from one with the first string being "}"
  if (not from.empty()) {
    to += '\0';
  }
  std::string converted;
  for (std::vector<std::string>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode_element(converted, *b)) {
      return false;
    }

    if (b != from.begin())
      to += ",";
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<string>

bool edm::encode_element(std::string& to, std::string const& from) { return encode(to, from); }

bool edm::encode_deprecated(std::string& to, std::vector<std::string> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<std::string>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode_deprecated(converted, *b)) {
      return false;
    }

    if (b != from.begin())
      to += ",";
    to += converted;
  }

  to += '}';
  return true;
}  // encode from vector<string>
// ----------------------------------------------------------------------
// ParameterSet
// ----------------------------------------------------------------------

bool edm::decode(ParameterSet& to, std::string_view from) {
  to = ParameterSet(from);
  return true;
}  // decode to ParameterSet

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, ParameterSet const& from) {
  to = from.toString();
  return true;
}  // encode from ParameterSet

// ----------------------------------------------------------------------

std::optional<std::string_view> edm::decode_pset_extent(std::string_view from) {
  auto e = ParameterSet::extent(from);
  if (e.empty()) {
    return {};
  }
  return e;
}

// ----------------------------------------------------------------------
// vPSet
// ----------------------------------------------------------------------
#include <iostream>
bool edm::decode(std::vector<ParameterSet>& to, std::string_view from) {
  to.clear();
  if (from.size() < 2) {
    return false;
  }
  if (from[0] != '{') {
    return false;
  }
  if (from.back() != '}') {
    return false;
  }

  to.reserve(std::count(from.begin(), from.end(), ',') + 1);

  auto remaining = from.substr(1, from.size() - 2);
  while (not remaining.empty()) {
    auto extent = ParameterSet::extent(remaining);
    if (extent.empty()) {
      return false;
    }
    ParameterSet val;
    if (!decode(val, extent)) {
      return false;
    }
    to.push_back(std::move(val));
    remaining.remove_prefix(extent.size());
    if (not remaining.empty()) {
      if (remaining[0] != ',') {
        return false;
      }
      remaining.remove_prefix(1);
    }
  }
  return true;
}  // decode to vector<ParameterSet>

// ----------------------------------------------------------------------

bool edm::encode(std::string& to, std::vector<ParameterSet> const& from) {
  to = "{";

  std::string converted;
  for (std::vector<ParameterSet>::const_iterator b = from.begin(), e = from.end(); b != e; ++b) {
    if (!encode(converted, *b)) {
      return false;
    }
    if (b != from.begin()) {
      to += ",";
    }
    to += converted;
  }
  to += '}';
  return true;
}  // encode from vector<ParameterSet>

// ----------------------------------------------------------------------

#include <iostream>
std::optional<std::string_view> edm::decode_vpset_extent(std::string_view from) {
  if (from.size() < 2) {
    return {};
  }
  if (from.front() != '{') {
    return {};
  }
  if (from[1] == '}') {
    return from.substr(0, 2);
  }
  if (from.size() < 3) {
    return {};
  }
  auto remaining = from.substr(1);
  while (not remaining.empty()) {
    auto extent = decode_pset_extent(remaining);
    if (not extent) {
      return {};
    }
    remaining.remove_prefix(extent->size());
    if (remaining.empty())
      return {};
    if (remaining.front() == '}') {
      return from.substr(0, from.size() - remaining.size() + 1);
    }
    if (remaining.front() == ',') {
      remaining = remaining.substr(1);
    } else {
      return {};
    }
  }
  return {};
}
