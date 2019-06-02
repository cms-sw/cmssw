// ----------------------------------------------------------------------
// definition of Entry's function members
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/Utilities/interface/Digest.h"

#include <map>
#include <sstream>
#include <ostream>
#include <cassert>
#include <iostream>

enum : char {
  kTbool = 'B',
  kTvBool = 'b',
  kTint32 = 'I',
  kTvint32 = 'i',
  kTuint32 = 'U',
  kTvuint32 = 'u',
  kTint64 = 'L',
  kTvint64 = 'l',
  kTuint64 = 'X',
  kTvuint64 = 'x',
  kTstring = 'S',
  kTvstring = 's',
  kTdouble = 'D',
  kTvdouble = 'd',
  kTPSet = 'P',
  kTvPSet = 'p',
  kTpath = 'T',
  kTFileInPath = 'F',
  kTInputTag = 't',
  kTVInputTag = 'v',
  kTESInputTag = 'g',
  kTVESInputTag = 'G',
  kTEventID = 'E',
  kTVEventID = 'e',
  kTLuminosityBlockID = 'M',
  kTVLuminosityBlockID = 'm',
  kTLuminosityBlockRange = 'A',
  kTVLuminosityBlockRange = 'a',
  kTEventRange = 'R',
  kTVEventRange = 'r'
};

namespace edm {
  namespace pset {

    struct TypeTrans {
      TypeTrans();

      typedef std::vector<std::string> CodeMap;
      CodeMap table_;
      std::map<std::string, char> type2Code_;
    };

    TypeTrans::TypeTrans() : table_(255) {
      table_[kTvBool] = "vBool";
      table_[kTbool] = "bool";
      table_[kTvint32] = "vint32";
      table_[kTint32] = "int32";
      table_[kTvuint32] = "vuint32";
      table_[kTuint32] = "uint32";
      table_[kTvint64] = "vint64";
      table_[kTint64] = "int64";
      table_[kTvuint64] = "vuint64";
      table_[kTuint64] = "uint64";
      table_[kTvstring] = "vstring";
      table_[kTstring] = "string";
      table_[kTvdouble] = "vdouble";
      table_[kTdouble] = "double";
      table_[kTvPSet] = "vPSet";
      table_[kTPSet] = "PSet";
      table_[kTpath] = "path";
      table_[kTFileInPath] = "FileInPath";
      table_[kTInputTag] = "InputTag";
      table_[kTVInputTag] = "VInputTag";
      table_[kTESInputTag] = "ESInputTag";
      table_[kTVESInputTag] = "VESInputTag";
      table_[kTVEventID] = "VEventID";
      table_[kTEventID] = "EventID";
      table_[kTVLuminosityBlockID] = "VLuminosityBlockID";
      table_[kTLuminosityBlockID] = "LuminosityBlockID";
      table_[kTVLuminosityBlockRange] = "VLuminosityBlockRange";
      table_[kTLuminosityBlockRange] = "LuminosityBlockRange";
      table_[kTVEventRange] = "VEventRange";
      table_[kTEventRange] = "EventRange";

      for (CodeMap::const_iterator itCode = table_.begin(), itCodeEnd = table_.end(); itCode != itCodeEnd; ++itCode) {
        type2Code_[*itCode] = (itCode - table_.begin());
      }
    }
  }  // namespace pset

  static pset::TypeTrans const sTypeTranslations;
  typedef std::map<std::string, char> Type2Code;

  // ----------------------------------------------------------------------
  // consistency-checker
  // ----------------------------------------------------------------------

  void Entry::validate() const {
    // tracked
    assert(tracked == '+' || tracked == '-');
    //     if(tracked != '+' && tracked != '-')
    //       throw EntryError(std::string("invalid tracked code ") + tracked);

    // type and rep
    switch (type) {
      case kTbool: {  // Bool
        bool val;
        if (!decode(val, rep))
          throwEntryError("bool", rep);
        break;
      }
      case kTvBool: {  // vBool
        std::vector<bool> val;
        if (!decode(val, rep))
          throwEntryError("vector<bool>", rep);
        break;
      }
      case kTint32: {  // Int32
        int val;
        if (!decode(val, rep))
          throwEntryError("int", rep);
        break;
      }
      case kTvint32: {  // vInt32
        std::vector<int> val;
        if (!decode(val, rep))
          throwEntryError("vector<int>", rep);
        break;
      }
      case kTuint32: {  // Uint32
        unsigned val;
        if (!decode(val, rep))
          throwEntryError("unsigned int", rep);
        break;
      }
      case kTvuint32: {  // vUint32
        std::vector<unsigned> val;
        if (!decode(val, rep))
          throwEntryError("vector<unsigned int>", rep);
        break;
      }
      case kTint64: {  // Int64
        int val;
        if (!decode(val, rep))
          throwEntryError("int64", rep);
        break;
      }
      case kTvint64: {  // vInt64
        std::vector<int> val;
        if (!decode(val, rep))
          throwEntryError("vector<int64>", rep);
        break;
      }
      case kTuint64: {  // Uint64
        unsigned val;
        if (!decode(val, rep))
          throwEntryError("unsigned int64", rep);
        break;
      }
      case kTvuint64: {  // vUint64
        std::vector<unsigned> val;
        if (!decode(val, rep))
          throwEntryError("vector<unsigned int64>", rep);
        break;
      }
      case kTstring: {  // String
        std::string val;
        if (!decode(val, rep))
          throwEntryError("string", rep);
        break;
      }
      case kTvstring: {  // vString
        std::vector<std::string> val;
        if (!decode(val, rep))
          throwEntryError("vector<string>", rep);
        break;
      }
      case kTFileInPath: {  // FileInPath
        FileInPath val;
        if (!decode(val, rep))
          throwEntryError("FileInPath", rep);
        break;
      }
      case kTInputTag: {  // InputTag
        InputTag val;
        if (!decode(val, rep))
          throwEntryError("InputTag", rep);
        break;
      }
      case kTVInputTag: {  // VInputTag
        std::vector<InputTag> val;
        if (!decode(val, rep))
          throwEntryError("VInputTag", rep);
        break;
      }
      case kTESInputTag: {  //ESInputTag
        ESInputTag val;
        if (!decode(val, rep))
          throwEntryError("ESInputTag", rep);
        break;
      }
      case kTVESInputTag: {  //ESInputTag
        std::vector<ESInputTag> val;
        if (!decode(val, rep))
          throwEntryError("VESInputTag", rep);
        break;
      }
      case kTEventID: {  // EventID
        EventID val;
        if (!decode(val, rep))
          throwEntryError("EventID", rep);
        break;
      }
      case kTVEventID: {  // VEventID
        std::vector<EventID> val;
        if (!decode(val, rep))
          throwEntryError("VEventID", rep);
        break;
      }
      case kTLuminosityBlockID: {  // LuminosityBlockID
        LuminosityBlockID val;
        if (!decode(val, rep))
          throwEntryError("LuminosityBlockID", rep);
        break;
      }
      case kTVLuminosityBlockID: {  // VLuminosityBlockID
        std::vector<LuminosityBlockID> val;
        if (!decode(val, rep))
          throwEntryError("VLuminosityBlockID", rep);
        break;
      }
      case kTdouble: {  // Double
        double val;
        if (!decode(val, rep))
          throwEntryError("double", rep);
        break;
      }
      case kTvdouble: {  // vDouble
        std::vector<double> val;
        if (!decode(val, rep))
          throwEntryError("vector<double>", rep);
        break;
      }
      case kTPSet: {  // ParameterSet
        ParameterSet val;
        if (!decode(val, rep))
          throwEntryError("ParameterSet", rep);
        break;
      }
      case kTvPSet: {  // vParameterSet
        std::vector<ParameterSet> val;
        if (!decode(val, rep))
          throwEntryError("vector<ParameterSet>", rep);
        break;
      }
      case kTLuminosityBlockRange: {  // LuminosityBlockRange
        LuminosityBlockRange val;
        if (!decode(val, rep))
          throwEntryError("LuminosityBlockRange", rep);
        break;
      }
      case kTVLuminosityBlockRange: {  // VLuminosityBlockRange
        std::vector<LuminosityBlockRange> val;
        if (!decode(val, rep))
          throwEntryError("VLuminosityBlockRange", rep);
        break;
      }
      case kTEventRange: {  // EventRange
        EventRange val;
        if (!decode(val, rep))
          throwEntryError("EventRange", rep);
        break;
      }
      case kTVEventRange: {  // VEventRange
        std::vector<EventRange> val;
        if (!decode(val, rep))
          throwEntryError("VEventRange", rep);
        break;
      }
      default: {
        // We should never get here.
        assert("Invalid type code" == nullptr);
        //throw EntryError(std::string("invalid type code ") + type);
        break;
      }
    }  // switch(type)
  }    // Entry::validate()

  // ----------------------------------------------------------------------
  // constructors
  // ----------------------------------------------------------------------

  // ----------------------------------------------------------------------
  // Bool

  Entry::Entry(std::string const& name, bool val, bool is_tracked)
      : name_(name), rep(), type(kTbool), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("bool");
    validate();
  }

  // ----------------------------------------------------------------------
  // Int32

  Entry::Entry(std::string const& name, int val, bool is_tracked)
      : name_(name), rep(), type(kTint32), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("int");
    validate();
  }

  // ----------------------------------------------------------------------
  // vInt32

  Entry::Entry(std::string const& name, std::vector<int> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvint32), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<int>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Uint32

  Entry::Entry(std::string const& name, unsigned val, bool is_tracked)
      : name_(name), rep(), type(kTuint32), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("unsigned int");
    validate();
  }

  // ----------------------------------------------------------------------
  // vUint32

  Entry::Entry(std::string const& name, std::vector<unsigned> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvuint32), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<unsigned int>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Int64

  Entry::Entry(std::string const& name, long long val, bool is_tracked)
      : name_(name), rep(), type(kTint64), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("int64");
    validate();
  }

  // ----------------------------------------------------------------------
  // vInt64

  Entry::Entry(std::string const& name, std::vector<long long> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvint64), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<int64>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Uint64

  Entry::Entry(std::string const& name, unsigned long long val, bool is_tracked)
      : name_(name), rep(), type(kTuint64), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("unsigned int64");
    validate();
  }

  // ----------------------------------------------------------------------
  // vUint64

  Entry::Entry(std::string const& name, std::vector<unsigned long long> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvuint64), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<unsigned int64>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Double

  Entry::Entry(std::string const& name, double val, bool is_tracked)
      : name_(name), rep(), type(kTdouble), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("double");
    validate();
  }

  // ----------------------------------------------------------------------
  // vDouble

  Entry::Entry(std::string const& name, std::vector<double> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvdouble), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<double>");
    validate();
  }

  // ----------------------------------------------------------------------
  // String

  Entry::Entry(std::string const& name, std::string const& val, bool is_tracked)
      : name_(name), rep(), type(kTstring), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("string");
    validate();
  }

  // ----------------------------------------------------------------------
  // vString

  Entry::Entry(std::string const& name, std::vector<std::string> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvstring), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<string>");
    validate();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  Entry::Entry(std::string const& name, FileInPath const& val, bool is_tracked)
      : name_(name), rep(), type(kTFileInPath), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("FileInPath");
    validate();
  }

  // ----------------------------------------------------------------------
  // InputTag

  Entry::Entry(std::string const& name, InputTag const& val, bool is_tracked)
      : name_(name), rep(), type(kTInputTag), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("InputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // VInputTag

  Entry::Entry(std::string const& name, std::vector<InputTag> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVInputTag), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VInputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  Entry::Entry(std::string const& name, ESInputTag const& val, bool is_tracked)
      : name_(name), rep(), type(kTESInputTag), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("InputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  Entry::Entry(std::string const& name, std::vector<ESInputTag> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVESInputTag), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VESInputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  //  EventID

  Entry::Entry(std::string const& name, EventID const& val, bool is_tracked)
      : name_(name), rep(), type(kTEventID), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("EventID");
    validate();
  }

  // ----------------------------------------------------------------------
  // VEventID

  Entry::Entry(std::string const& name, std::vector<EventID> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVEventID), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VEventID");
    validate();
  }

  // ----------------------------------------------------------------------
  //  LuminosityBlockID

  Entry::Entry(std::string const& name, LuminosityBlockID const& val, bool is_tracked)
      : name_(name), rep(), type(kTLuminosityBlockID), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("LuminosityBlockID");
    validate();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  Entry::Entry(std::string const& name, std::vector<LuminosityBlockID> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVLuminosityBlockID), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VLuminosityBlockID");
    validate();
  }

  // ----------------------------------------------------------------------
  //  LuminosityBlockRange

  Entry::Entry(std::string const& name, LuminosityBlockRange const& val, bool is_tracked)
      : name_(name), rep(), type(kTLuminosityBlockRange), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("LuminosityBlockRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  Entry::Entry(std::string const& name, std::vector<LuminosityBlockRange> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVLuminosityBlockRange), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VLuminosityBlockRange");
    validate();
  }

  // ----------------------------------------------------------------------
  //  EventRange

  Entry::Entry(std::string const& name, EventRange const& val, bool is_tracked)
      : name_(name), rep(), type(kTEventRange), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("EventRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // VEventRange

  Entry::Entry(std::string const& name, std::vector<EventRange> const& val, bool is_tracked)
      : name_(name), rep(), type(kTVEventRange), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("VEventRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // ParameterSet

  Entry::Entry(std::string const& name, ParameterSet const& val, bool is_tracked)
      : name_(name), rep(), type(kTPSet), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("ParameterSet");
    validate();
  }

  // ----------------------------------------------------------------------
  // vPSet

  Entry::Entry(std::string const& name, std::vector<ParameterSet> const& val, bool is_tracked)
      : name_(name), rep(), type(kTvPSet), tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throwEncodeError("vector<ParameterSet>");
    validate();
  }

  // ----------------------------------------------------------------------
  // coded string

  Entry::Entry(std::string const& name, std::string const& code) : name_(name), rep(), type('?'), tracked('?') {
    if (!fromString(code.begin(), code.end()))
      throwEncodeError("coded string");
    validate();
  }

  Entry::Entry(std::string const& name, std::string const& type, std::string const& value, bool is_tracked)
      : name_(name), rep(), type('?'), tracked('?') {
    std::string codedString(is_tracked ? "-" : "+");

    Type2Code::const_iterator itFound = sTypeTranslations.type2Code_.find(type);
    if (itFound == sTypeTranslations.type2Code_.end()) {
      throw Exception(errors::Configuration) << "bad type name used for Entry : " << type;
    }

    codedString += itFound->second;
    codedString += '(';
    codedString += value;
    codedString += ')';

    if (!fromString(codedString.begin(), codedString.end())) {
      throw Exception(errors::Configuration) << "bad encoded Entry string " << codedString;
    }
    validate();
  }

  Entry::Entry(std::string const& name, std::string const& type, std::vector<std::string> const& value, bool is_tracked)
      : name_(name), rep(), type('?'), tracked('?') {
    std::string codedString(is_tracked ? "-" : "+");

    Type2Code::const_iterator itFound = sTypeTranslations.type2Code_.find(type);
    if (itFound == sTypeTranslations.type2Code_.end()) {
      throw Exception(errors::Configuration) << "bad type name used for Entry : " << type;
    }

    codedString += itFound->second;
    codedString += '(';
    codedString += '{';
    std::vector<std::string>::const_iterator i = value.begin();
    std::vector<std::string>::const_iterator e = value.end();
    std::string const kSeparator(",");
    std::string sep("");
    for (; i != e; ++i) {
      codedString += sep;
      codedString += *i;
      sep = kSeparator;
    }
    codedString += '}';
    codedString += ')';

    if (!fromString(codedString.begin(), codedString.end())) {
      throw Exception(errors::Configuration) << "bad encoded Entry string " << codedString;
    }
    validate();
  }

  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------

  void Entry::toString(std::string& result) const {
    result.reserve(result.size() + sizeOfString());
    result += tracked;
    result += type;
    result += '(';
    result += rep;
    result += ')';
  }

  void Entry::toDigest(cms::Digest& digest) const {
    digest.append(&tracked, 1);
    digest.append(&type, 1);
    digest.append("(", 1);
    digest.append(rep);
    digest.append(")", 1);
  }

  std::string Entry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  // ----------------------------------------------------------------------

  bool Entry::fromString(std::string::const_iterator const b, std::string::const_iterator const e) {
    if (static_cast<unsigned int>(e - b) < 4u || b[2] != '(' || e[-1] != ')')

      return false;

    tracked = b[0];
    type = b[1];
    rep = std::string(b + 3, e - 1);

    return true;
  }  // from_string()

  // ----------------------------------------------------------------------
  // value accessors
  // ----------------------------------------------------------------------

  // ----------------------------------------------------------------------
  // Bool

  bool Entry::getBool() const {
    if (type != kTbool)
      throwValueError("bool");
    bool val;
    if (!decode(val, rep))
      throwEntryError("bool", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // Int32

  int Entry::getInt32() const {
    if (type != kTint32)
      throwValueError("int");
    int val;
    if (!decode(val, rep))
      throwEntryError("int", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vInt32

  std::vector<int> Entry::getVInt32() const {
    if (type != kTvint32)
      throwValueError("vector<int>");
    std::vector<int> val;
    if (!decode(val, rep))
      throwEntryError("vector<int>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // Int32

  long long Entry::getInt64() const {
    if (type != kTint64)
      throwValueError("int64");
    long long val;
    if (!decode(val, rep))
      throwEntryError("int64", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vInt32

  std::vector<long long> Entry::getVInt64() const {
    if (type != kTvint64)
      throwValueError("vector<int64>");
    std::vector<long long> val;
    if (!decode(val, rep))
      throwEntryError("vector<int64>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // Uint32

  unsigned Entry::getUInt32() const {
    if (type != kTuint32)
      throwValueError("unsigned int");
    unsigned val;
    if (!decode(val, rep))
      throwEntryError("unsigned int", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vUint32

  std::vector<unsigned> Entry::getVUInt32() const {
    if (type != kTvuint32)
      throwValueError("vector<unsigned int>");
    std::vector<unsigned> val;
    if (!decode(val, rep))
      throwEntryError("vector<unsigned int>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // Uint64

  unsigned long long Entry::getUInt64() const {
    if (type != kTuint64)
      throwValueError("uint64");
    unsigned long long val;
    if (!decode(val, rep))
      throwEntryError("uint64", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vUint64

  std::vector<unsigned long long> Entry::getVUInt64() const {
    if (type != kTvuint64)
      throwValueError("vector<uint64>");
    std::vector<unsigned long long> val;
    if (!decode(val, rep))
      throwEntryError("vector<uint64>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // Double

  double Entry::getDouble() const {
    if (type != kTdouble)
      throwValueError("double");
    double val;
    if (!decode(val, rep))
      throwEntryError("double", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vDouble

  std::vector<double> Entry::getVDouble() const {
    if (type != kTvdouble)
      throwValueError("vector<double>");
    std::vector<double> val;
    if (!decode(val, rep))
      throwEntryError("vector<double>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // String

  std::string Entry::getString() const {
    if (type != kTstring)
      throwValueError("string");
    std::string val;
    if (!decode(val, rep))
      throwEntryError("string", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vString

  std::vector<std::string> Entry::getVString() const {
    if (type != kTvstring)
      throwValueError("vector<string>");
    std::vector<std::string> val;
    if (!decode(val, rep))
      throwEntryError("vector<string>", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // FileInPath

  FileInPath Entry::getFileInPath() const {
    if (type != kTFileInPath)
      throwValueError("FileInPath");
    FileInPath val;
    if (!decode(val, rep))
      throwEntryError("FileInPath", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // InputTag

  InputTag Entry::getInputTag() const {
    if (type != kTInputTag)
      throwValueError("InputTag");
    InputTag val;
    if (!decode(val, rep))
      throwEntryError("InputTag", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VInputTag

  std::vector<InputTag> Entry::getVInputTag() const {
    if (type != kTVInputTag)
      throwValueError("VInputTag");
    std::vector<InputTag> val;
    if (!decode(val, rep))
      throwEntryError("VInputTag", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  ESInputTag Entry::getESInputTag() const {
    if (type != kTESInputTag)
      throwValueError("ESInputTag");
    ESInputTag val;
    if (!decode(val, rep))
      throwEntryError("ESInputTag", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  std::vector<ESInputTag> Entry::getVESInputTag() const {
    if (type != kTVESInputTag)
      throwValueError("VESInputTag");
    std::vector<ESInputTag> val;
    if (!decode(val, rep))
      throwEntryError("VESInputTag", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // EventID

  EventID Entry::getEventID() const {
    if (type != kTEventID)
      throwValueError("EventID");
    EventID val;
    if (!decode(val, rep))
      throwEntryError("EventID", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VEventID

  std::vector<EventID> Entry::getVEventID() const {
    if (type != kTVEventID)
      throwValueError("VEventID");
    std::vector<EventID> val;
    if (!decode(val, rep))
      throwEntryError("EventID", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  LuminosityBlockID Entry::getLuminosityBlockID() const {
    if (type != kTLuminosityBlockID)
      throwValueError("LuminosityBlockID");
    LuminosityBlockID val;
    if (!decode(val, rep))
      throwEntryError("LuminosityBlockID", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  std::vector<LuminosityBlockID> Entry::getVLuminosityBlockID() const {
    if (type != kTVLuminosityBlockID)
      throwValueError("VLuminosityBlockID");
    std::vector<LuminosityBlockID> val;
    if (!decode(val, rep))
      throwEntryError("LuminosityBlockID", rep);
    return val;
  }

  // LuminosityBlockRange

  LuminosityBlockRange Entry::getLuminosityBlockRange() const {
    if (type != kTLuminosityBlockRange)
      throwValueError("LuminosityBlockRange");
    LuminosityBlockRange val;
    if (!decode(val, rep))
      throwEntryError("LuminosityBlockRange", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  std::vector<LuminosityBlockRange> Entry::getVLuminosityBlockRange() const {
    if (type != kTVLuminosityBlockRange)
      throwValueError("VLuminosityBlockRange");
    std::vector<LuminosityBlockRange> val;
    if (!decode(val, rep))
      throwEntryError("LuminosityBlockRange", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // EventRange

  EventRange Entry::getEventRange() const {
    if (type != kTEventRange)
      throwValueError("EventRange");
    EventRange val;
    if (!decode(val, rep))
      throwEntryError("EventRange", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // VEventRange

  std::vector<EventRange> Entry::getVEventRange() const {
    if (type != kTVEventRange)
      throwValueError("VEventRange");
    std::vector<EventRange> val;
    if (!decode(val, rep))
      throwEntryError("EventRange", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // ParameterSet

  ParameterSet Entry::getPSet() const {
    if (type != kTPSet)
      throwValueError("ParameterSet");
    ParameterSet val;
    if (!decode(val, rep))
      throwEntryError("ParameterSet", rep);
    return val;
  }

  // ----------------------------------------------------------------------
  // vPSet

  std::vector<ParameterSet> Entry::getVPSet() const {
    if (type != kTvPSet)
      throwValueError("vector<ParameterSet>");
    std::vector<ParameterSet> val;
    if (!decode(val, rep))
      throwEntryError("vector<ParameterSet>", rep);
    return val;
  }

  std::ostream& operator<<(std::ostream& os, Entry const& entry) {
    os << sTypeTranslations.table_[entry.typeCode()] << " " << (entry.isTracked() ? "tracked " : "untracked ") << " = ";

    // now handle the difficult cases
    switch (entry.typeCode()) {
      case kTPSet:  // ParameterSet
      {
        os << entry.getPSet();
        break;
      }
      case 'p':  // vector<ParameterSet>
      {
        // Make sure we get the representation of each contained
        // ParameterSet including *only* tracked parameters
        std::vector<ParameterSet> whole = entry.getVPSet();
        std::vector<ParameterSet>::const_iterator i = whole.begin();
        std::vector<ParameterSet>::const_iterator e = whole.end();
        std::string start = "";
        std::string const between(",\n");
        os << "{" << std::endl;
        for (; i != e; ++i) {
          os << start << *i;
          start = between;
        }
        if (!whole.empty()) {
          os << std::endl;
        }
        os << "}";
        break;
      }
      case kTstring: {
        os << "'" << entry.getString() << "'";
        break;
      }
      case kTvstring: {
        os << "{";
        std::string start = "'";
        std::string const between(",'");
        std::vector<std::string> strings = entry.getVString();
        for (std::vector<std::string>::const_iterator it = strings.begin(), itEnd = strings.end(); it != itEnd; ++it) {
          os << start << *it << "'";
          start = between;
        }
        os << "}";
        break;
      }
      case kTint32: {
        os << entry.getInt32();
        break;
      }
      case kTuint32: {
        os << entry.getUInt32();
        break;
      }
      case kTVInputTag: {
        //VInputTag needs to be treated seperately because it is encode like
        // vector<string> rather than using the individual encodings of each InputTag
        os << "{";
        std::string start = "";
        std::string const between(",");
        std::vector<InputTag> tags = entry.getVInputTag();
        for (std::vector<InputTag>::const_iterator it = tags.begin(), itEnd = tags.end(); it != itEnd; ++it) {
          os << start << it->encode();
          start = between;
        }
        os << "}";
        break;
      }
      case kTVESInputTag: {
        //VESInputTag needs to be treated seperately because it is encode like
        // vector<string> rather than using the individual encodings of each ESInputTag
        os << "{";
        std::string start = "";
        std::string const between(",");
        std::vector<ESInputTag> tags = entry.getVESInputTag();
        for (std::vector<ESInputTag>::const_iterator it = tags.begin(), itEnd = tags.end(); it != itEnd; ++it) {
          os << start << it->encode();
          start = between;
        }
        os << "}";
        break;
      }
      default: {
        os << entry.rep;
        break;
      }
    }

    return os;
  }

  // Helper functions for throwing exceptions

  void Entry::throwValueError(char const* expectedType) const {
    throw Exception(errors::Configuration, "ValueError") << "type of " << name_ << " is expected to be " << expectedType
                                                         << " but declared as " << sTypeTranslations.table_[type];
  }

  void Entry::throwEntryError(char const* expectedType, std::string const& badRep) const {
    throw Exception(errors::Configuration, "EntryError") << "can not convert representation of " << name_ << ": "
                                                         << badRep << " to value of type " << expectedType << " ";
  }

  void Entry::throwEncodeError(char const* type) const {
    throw Exception(errors::Configuration, "EncodingError") << "can not encode " << name_ << " as type: " << type;
  }

}  // namespace edm
