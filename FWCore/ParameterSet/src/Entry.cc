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
#include <string_view>

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
  kTstringHex = 'S',
  kTvstringHex = 's',
  kTstringRaw = 'Z',
  kTvstringRaw = 'z',
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

static constexpr const std::array<std::string_view, 32> s_types = {{"bool",
                                                                    "double",
                                                                    "int32",
                                                                    "int64",
                                                                    "path",
                                                                    "string",
                                                                    "deprecated_string",
                                                                    "uint32",
                                                                    "uint64",
                                                                    "vdouble",
                                                                    "vint32",
                                                                    "vint64",
                                                                    "vstring",
                                                                    "vstring_deprecated",
                                                                    "vuint32",
                                                                    "vuint64",
                                                                    "vBool",
                                                                    "vPSet",
                                                                    "EventID",
                                                                    "EventRange",
                                                                    "ESInputTag",
                                                                    "FileInPath",
                                                                    "InputTag",
                                                                    "LuminosityBlockID",
                                                                    "LuminosityBlockRange",
                                                                    "PSet",
                                                                    "VEventID",
                                                                    "VEventRange",
                                                                    "VESInputTag",
                                                                    "VInputTag",
                                                                    "VLuminosityBlockID",
                                                                    "VLuminosityBlockRange"}};

static constexpr const std::array<char, 32> s_codes = {{kTbool,
                                                        kTdouble,
                                                        kTint32,
                                                        kTint64,
                                                        kTpath,
                                                        kTstringRaw,
                                                        kTstringHex,
                                                        kTuint32,
                                                        kTuint64,
                                                        kTvdouble,
                                                        kTvint32,
                                                        kTvint64,
                                                        kTvstringRaw,
                                                        kTvstringHex,
                                                        kTvuint32,
                                                        kTvuint64,
                                                        kTvBool,
                                                        kTvPSet,
                                                        kTEventID,
                                                        kTEventRange,
                                                        kTESInputTag,
                                                        kTFileInPath,
                                                        kTInputTag,
                                                        kTLuminosityBlockID,
                                                        kTLuminosityBlockRange,
                                                        kTPSet,
                                                        kTVEventID,
                                                        kTVEventRange,
                                                        kTVESInputTag,
                                                        kTVInputTag,
                                                        kTVLuminosityBlockID,
                                                        kTVLuminosityBlockRange}};

//a compile time function to convert code to type
// not used at runtime since does linear search
static constexpr std::string_view c2t(char iCode) {
  static_assert(s_codes.size() == s_types.size());
  for (size_t index = 0; index < s_codes.size(); ++index) {
    if (s_codes[index] == iCode) {
      return s_types[index];
    }
  }
  return std::string_view();
}

static constexpr std::array<std::string_view, 255> fillTable() {
  std::array<std::string_view, 255> table_ = {{std::string_view()}};
  static_assert(not c2t(kTvBool).empty());
  table_[kTvBool] = c2t(kTvBool);
  static_assert(not c2t(kTbool).empty());
  table_[kTbool] = c2t(kTbool);
  static_assert(not c2t(kTvint32).empty());
  table_[kTvint32] = c2t(kTvint32);
  static_assert(not c2t(kTint32).empty());
  table_[kTint32] = c2t(kTint32);
  static_assert(not c2t(kTvuint32).empty());
  table_[kTvuint32] = c2t(kTvuint32);
  static_assert(not c2t(kTuint32).empty());
  table_[kTuint32] = c2t(kTuint32);
  static_assert(not c2t(kTvint64).empty());
  table_[kTvint64] = c2t(kTvint64);
  static_assert(not c2t(kTint64).empty());
  table_[kTint64] = c2t(kTint64);
  static_assert(not c2t(kTvuint64).empty());
  table_[kTvuint64] = c2t(kTvuint64);
  static_assert(not c2t(kTuint64).empty());
  table_[kTuint64] = c2t(kTuint64);
  static_assert(not c2t(kTvstringRaw).empty());
  table_[kTvstringRaw] = c2t(kTvstringRaw);
  static_assert(not c2t(kTvstringHex).empty());
  table_[kTvstringHex] = c2t(kTvstringHex);
  static_assert(not c2t(kTstringRaw).empty());
  table_[kTstringRaw] = c2t(kTstringRaw);
  static_assert(not c2t(kTstringHex).empty());
  table_[kTstringHex] = c2t(kTstringHex);
  static_assert(not c2t(kTvdouble).empty());
  table_[kTvdouble] = c2t(kTvdouble);
  static_assert(not c2t(kTdouble).empty());
  table_[kTdouble] = c2t(kTdouble);
  static_assert(not c2t(kTvPSet).empty());
  table_[kTvPSet] = c2t(kTvPSet);
  static_assert(not c2t(kTPSet).empty());
  table_[kTPSet] = c2t(kTPSet);
  static_assert(not c2t(kTpath).empty());
  table_[kTpath] = c2t(kTpath);
  static_assert(not c2t(kTFileInPath).empty());
  table_[kTFileInPath] = c2t(kTFileInPath);
  static_assert(not c2t(kTInputTag).empty());
  table_[kTInputTag] = c2t(kTInputTag);
  static_assert(not c2t(kTVInputTag).empty());
  table_[kTVInputTag] = c2t(kTVInputTag);
  static_assert(not c2t(kTESInputTag).empty());
  table_[kTESInputTag] = c2t(kTESInputTag);
  static_assert(not c2t(kTVESInputTag).empty());
  table_[kTVESInputTag] = c2t(kTVESInputTag);
  static_assert(not c2t(kTVEventID).empty());
  table_[kTVEventID] = c2t(kTVEventID);
  static_assert(not c2t(kTFileInPath).empty());
  table_[kTEventID] = c2t(kTEventID);
  static_assert(not c2t(kTVLuminosityBlockID).empty());
  table_[kTVLuminosityBlockID] = c2t(kTVLuminosityBlockID);
  static_assert(not c2t(kTLuminosityBlockID).empty());
  table_[kTLuminosityBlockID] = c2t(kTLuminosityBlockID);
  static_assert(not c2t(kTVLuminosityBlockRange).empty());
  table_[kTVLuminosityBlockRange] = c2t(kTVLuminosityBlockRange);
  static_assert(not c2t(kTLuminosityBlockRange).empty());
  table_[kTLuminosityBlockRange] = c2t(kTLuminosityBlockRange);
  static_assert(not c2t(kTVEventRange).empty());
  table_[kTVEventRange] = c2t(kTVEventRange);
  static_assert(not c2t(kTEventRange).empty());
  table_[kTEventRange] = c2t(kTEventRange);
  return table_;
}

static constexpr const std::array<std::string_view, 255> s_table = fillTable();

static constexpr std::string_view typeFromCode(char iCode) { return s_table[iCode]; }

static char codeFromType(std::string_view iType) {
  auto itFound = std::lower_bound(s_types.begin(), s_types.end(), iType);
  if (itFound == s_types.end() or *itFound != iType) {
    throw edm::Exception(edm::errors::Configuration) << "bad type name used for Entry : " << iType;
  }
  return s_codes[itFound - s_types.begin()];
}
namespace edm {
  // ----------------------------------------------------------------------
  // consistency-checker
  // ----------------------------------------------------------------------

  void Entry::validate() const {
    // tracked
    assert(tracked_ == '+' || tracked_ == '-');
    //     if(tracked_ != '+' && tracked_ != '-')
    //       throw EntryError(std::string("invalid tracked code ") + tracked_);

    // type and rep
    switch (type_) {
      case kTbool: {  // Bool
        bool val;
        if (!decode(val, rep_))
          throwEntryError("bool", rep_);
        break;
      }
      case kTvBool: {  // vBool
        std::vector<bool> val;
        if (!decode(val, rep_))
          throwEntryError("vector<bool>", rep_);
        break;
      }
      case kTint32: {  // Int32
        int val;
        if (!decode(val, rep_))
          throwEntryError("int", rep_);
        break;
      }
      case kTvint32: {  // vInt32
        std::vector<int> val;
        if (!decode(val, rep_))
          throwEntryError("vector<int>", rep_);
        break;
      }
      case kTuint32: {  // Uint32
        unsigned val;
        if (!decode(val, rep_))
          throwEntryError("unsigned int", rep_);
        break;
      }
      case kTvuint32: {  // vUint32
        std::vector<unsigned> val;
        if (!decode(val, rep_))
          throwEntryError("vector<unsigned int>", rep_);
        break;
      }
      case kTint64: {  // Int64
        long long int val;
        if (!decode(val, rep_))
          throwEntryError("int64", rep_);
        break;
      }
      case kTvint64: {  // vInt64
        std::vector<long long int> val;
        if (!decode(val, rep_))
          throwEntryError("vector<int64>", rep_);
        break;
      }
      case kTuint64: {  // Uint64
        unsigned long long int val;
        if (!decode(val, rep_))
          throwEntryError("unsigned int64", rep_);
        break;
      }
      case kTvuint64: {  // vUint64
        std::vector<unsigned long long int> val;
        if (!decode(val, rep_))
          throwEntryError("vector<unsigned int64>", rep_);
        break;
      }
      case kTstringRaw: {  // String
        std::string val;
        if (!decode(val, rep_))
          throwEntryError("string", rep_);
        break;
      }
      case kTvstringRaw: {  // vString
        std::vector<std::string> val;
        if (!decode(val, rep_))
          throwEntryError("vector<string>", rep_);
        break;
      }
      case kTstringHex: {  // String
        std::string val;
        if (!decode_deprecated(val, rep_))
          throwEntryError("string_hex", rep_);
        break;
      }
      case kTvstringHex: {  // vString
        std::vector<std::string> val;
        if (!decode_deprecated(val, rep_))
          throwEntryError("vector<string_hex>", rep_);
        break;
      }

      case kTFileInPath: {  // FileInPath
        FileInPath val;
        if (!decode(val, rep_))
          throwEntryError("FileInPath", rep_);
        break;
      }
      case kTInputTag: {  // InputTag
        InputTag val;
        if (!decode(val, rep_))
          throwEntryError("InputTag", rep_);
        break;
      }
      case kTVInputTag: {  // VInputTag
        std::vector<InputTag> val;
        if (!decode(val, rep_))
          throwEntryError("VInputTag", rep_);
        break;
      }
      case kTESInputTag: {  //ESInputTag
        ESInputTag val;
        if (!decode(val, rep_))
          throwEntryError("ESInputTag", rep_);
        break;
      }
      case kTVESInputTag: {  //ESInputTag
        std::vector<ESInputTag> val;
        if (!decode(val, rep_))
          throwEntryError("VESInputTag", rep_);
        break;
      }
      case kTEventID: {  // EventID
        EventID val;
        if (!decode(val, rep_))
          throwEntryError("EventID", rep_);
        break;
      }
      case kTVEventID: {  // VEventID
        std::vector<EventID> val;
        if (!decode(val, rep_))
          throwEntryError("VEventID", rep_);
        break;
      }
      case kTLuminosityBlockID: {  // LuminosityBlockID
        LuminosityBlockID val;
        if (!decode(val, rep_))
          throwEntryError("LuminosityBlockID", rep_);
        break;
      }
      case kTVLuminosityBlockID: {  // VLuminosityBlockID
        std::vector<LuminosityBlockID> val;
        if (!decode(val, rep_))
          throwEntryError("VLuminosityBlockID", rep_);
        break;
      }
      case kTdouble: {  // Double
        double val;
        if (!decode(val, rep_))
          throwEntryError("double", rep_);
        break;
      }
      case kTvdouble: {  // vDouble
        std::vector<double> val;
        if (!decode(val, rep_))
          throwEntryError("vector<double>", rep_);
        break;
      }
      case kTPSet: {  // ParameterSet
        ParameterSet val;
        if (!decode(val, rep_))
          throwEntryError("ParameterSet", rep_);
        break;
      }
      case kTvPSet: {  // vParameterSet
        std::vector<ParameterSet> val;
        if (!decode(val, rep_))
          throwEntryError("vector<ParameterSet>", rep_);
        break;
      }
      case kTLuminosityBlockRange: {  // LuminosityBlockRange
        LuminosityBlockRange val;
        if (!decode(val, rep_))
          throwEntryError("LuminosityBlockRange", rep_);
        break;
      }
      case kTVLuminosityBlockRange: {  // VLuminosityBlockRange
        std::vector<LuminosityBlockRange> val;
        if (!decode(val, rep_))
          throwEntryError("VLuminosityBlockRange", rep_);
        break;
      }
      case kTEventRange: {  // EventRange
        EventRange val;
        if (!decode(val, rep_))
          throwEntryError("EventRange", rep_);
        break;
      }
      case kTVEventRange: {  // VEventRange
        std::vector<EventRange> val;
        if (!decode(val, rep_))
          throwEntryError("VEventRange", rep_);
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
      : name_(name), rep_(), type_(kTbool), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("bool");
    validate();
  }

  // ----------------------------------------------------------------------
  // Int32

  Entry::Entry(std::string const& name, int val, bool is_tracked)
      : name_(name), rep_(), type_(kTint32), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("int");
    validate();
  }

  // ----------------------------------------------------------------------
  // vInt32

  Entry::Entry(std::string const& name, std::vector<int> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvint32), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<int>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Uint32

  Entry::Entry(std::string const& name, unsigned val, bool is_tracked)
      : name_(name), rep_(), type_(kTuint32), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("unsigned int");
    validate();
  }

  // ----------------------------------------------------------------------
  // vUint32

  Entry::Entry(std::string const& name, std::vector<unsigned> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvuint32), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<unsigned int>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Int64

  Entry::Entry(std::string const& name, long long val, bool is_tracked)
      : name_(name), rep_(), type_(kTint64), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("int64");
    validate();
  }

  // ----------------------------------------------------------------------
  // vInt64

  Entry::Entry(std::string const& name, std::vector<long long> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvint64), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<int64>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Uint64

  Entry::Entry(std::string const& name, unsigned long long val, bool is_tracked)
      : name_(name), rep_(), type_(kTuint64), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("unsigned int64");
    validate();
  }

  // ----------------------------------------------------------------------
  // vUint64

  Entry::Entry(std::string const& name, std::vector<unsigned long long> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvuint64), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<unsigned int64>");
    validate();
  }

  // ----------------------------------------------------------------------
  // Double

  Entry::Entry(std::string const& name, double val, bool is_tracked)
      : name_(name), rep_(), type_(kTdouble), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("double");
    validate();
  }

  // ----------------------------------------------------------------------
  // vDouble

  Entry::Entry(std::string const& name, std::vector<double> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvdouble), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<double>");
    validate();
  }

  // ----------------------------------------------------------------------
  // String

  Entry::Entry(std::string const& name, std::string const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTstringRaw), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("string");
    validate();
  }

  // ----------------------------------------------------------------------
  // vString

  Entry::Entry(std::string const& name, std::vector<std::string> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvstringRaw), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<string>");
    validate();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  Entry::Entry(std::string const& name, FileInPath const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTFileInPath), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("FileInPath");
    validate();
  }

  // ----------------------------------------------------------------------
  // InputTag

  Entry::Entry(std::string const& name, InputTag const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTInputTag), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("InputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // VInputTag

  Entry::Entry(std::string const& name, std::vector<InputTag> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVInputTag), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VInputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  Entry::Entry(std::string const& name, ESInputTag const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTESInputTag), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("InputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  Entry::Entry(std::string const& name, std::vector<ESInputTag> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVESInputTag), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VESInputTag");
    validate();
  }

  // ----------------------------------------------------------------------
  //  EventID

  Entry::Entry(std::string const& name, EventID const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTEventID), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("EventID");
    validate();
  }

  // ----------------------------------------------------------------------
  // VEventID

  Entry::Entry(std::string const& name, std::vector<EventID> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVEventID), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VEventID");
    validate();
  }

  // ----------------------------------------------------------------------
  //  LuminosityBlockID

  Entry::Entry(std::string const& name, LuminosityBlockID const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTLuminosityBlockID), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("LuminosityBlockID");
    validate();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  Entry::Entry(std::string const& name, std::vector<LuminosityBlockID> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVLuminosityBlockID), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VLuminosityBlockID");
    validate();
  }

  // ----------------------------------------------------------------------
  //  LuminosityBlockRange

  Entry::Entry(std::string const& name, LuminosityBlockRange const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTLuminosityBlockRange), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("LuminosityBlockRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  Entry::Entry(std::string const& name, std::vector<LuminosityBlockRange> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVLuminosityBlockRange), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VLuminosityBlockRange");
    validate();
  }

  // ----------------------------------------------------------------------
  //  EventRange

  Entry::Entry(std::string const& name, EventRange const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTEventRange), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("EventRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // VEventRange

  Entry::Entry(std::string const& name, std::vector<EventRange> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTVEventRange), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("VEventRange");
    validate();
  }

  // ----------------------------------------------------------------------
  // ParameterSet

  Entry::Entry(std::string const& name, ParameterSet const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTPSet), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("ParameterSet");
    validate();
  }

  // ----------------------------------------------------------------------
  // vPSet

  Entry::Entry(std::string const& name, std::vector<ParameterSet> const& val, bool is_tracked)
      : name_(name), rep_(), type_(kTvPSet), tracked_(is_tracked ? '+' : '-') {
    if (!encode(rep_, val))
      throwEncodeError("vector<ParameterSet>");
    validate();
  }

  // ----------------------------------------------------------------------
  // coded string

  Entry::Entry(std::string name, std::string_view code) : name_(std::move(name)), rep_(), type_('?'), tracked_('?') {
    if (!fromString(code.begin(), code.end()))
      throwEncodeError("coded string");
    validate();
  }

  Entry::Entry(std::string name, std::string_view type, std::string_view value, bool is_tracked)
      : name_(std::move(name)), rep_(), type_('?'), tracked_('?') {
    std::string codedString(is_tracked ? "-" : "+");

    codedString += codeFromType(type);
    codedString += '(';
    codedString += value;
    codedString += ')';

    std::string_view v = codedString;
    if (!fromString(v.begin(), v.end())) {
      throw Exception(errors::Configuration) << "bad encoded Entry string " << codedString;
    }
    validate();
  }

  Entry::Entry(std::string name, std::string_view type, std::vector<std::string> const& value, bool is_tracked)
      : name_(std::move(name)), rep_(), type_('?'), tracked_('?') {
    std::string codedString(is_tracked ? "-" : "+");

    codedString += codeFromType(type);
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

    std::string_view v = codedString;
    if (!fromString(v.begin(), v.end())) {
      throw Exception(errors::Configuration) << "bad encoded Entry string " << codedString;
    }
    validate();
  }

  std::string_view Entry::bounds(std::string_view iView, std::size_t iEndHint) {
    if (iView.size() < 4) {
      return {};
    }
    if (iView[1] == kTPSet) {
      auto extent = edm::decode_pset_extent(iView.substr(3));
      if (not extent) {
        return {};
      }
      return iView.substr(0, extent->size() + 3 + 1);
    } else if (iView[1] == kTvPSet) {
      auto extent = edm::decode_vpset_extent(iView.substr(3));
      if (not extent) {
        return {};
      }
      return iView.substr(0, extent->size() + 3 + 1);
    } else if (iView[1] != kTstringRaw and iView[1] != kTvstringRaw) {
      return iView.substr(0, iEndHint);
    }
    if (iView[1] == kTstringRaw) {
      // is of the form
      // [trackiness][type](...\0)[endHintChar]
      // where decode returns the ...\0 part
      auto extent = edm::decode_string_extent(iView.substr(3));
      if (not extent) {
        return {};
      }
      return iView.substr(0, extent->size() + 3 + 1);
    }
    // is of the form
    // [trackiness][type]({...})[endHitChar]
    // where decode returns the {...} part
    auto extent = edm::decode_vstring_extent(iView.substr(3));
    if (not extent) {
      return {};
    }
    return iView.substr(0, extent->size() + 3 + 1);
  }

  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------

  void Entry::toString(std::string& result) const {
    result.reserve(result.size() + sizeOfString());
    result += tracked_;
    result += type_;
    result += '(';
    result += rep_;
    result += ')';
  }

  void Entry::toDigest(cms::Digest& digest) const {
    digest.append(&tracked_, 1);
    digest.append(&type_, 1);
    digest.append("(", 1);
    digest.append(rep_);
    digest.append(")", 1);
  }

  std::string Entry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

  // ----------------------------------------------------------------------

  bool Entry::fromString(std::string_view::const_iterator const b, std::string_view::const_iterator const e) {
    if (static_cast<unsigned int>(e - b) < 4u || b[2] != '(' || e[-1] != ')')

      return false;

    tracked_ = b[0];
    type_ = b[1];
    rep_ = std::string(b + 3, e - 1);

    return true;
  }  // from_string()

  // ----------------------------------------------------------------------
  // value accessors
  // ----------------------------------------------------------------------

  // ----------------------------------------------------------------------
  // Bool

  bool Entry::getBool() const {
    if (type_ != kTbool)
      throwValueError("bool");
    bool val;
    if (!decode(val, rep_))
      throwEntryError("bool", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // Int32

  int Entry::getInt32() const {
    if (type_ != kTint32)
      throwValueError("int");
    int val;
    if (!decode(val, rep_))
      throwEntryError("int", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vInt32

  std::vector<int> Entry::getVInt32() const {
    if (type_ != kTvint32)
      throwValueError("vector<int>");
    std::vector<int> val;
    if (!decode(val, rep_))
      throwEntryError("vector<int>", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // Int32

  long long Entry::getInt64() const {
    if (type_ != kTint64)
      throwValueError("int64");
    long long val;
    if (!decode(val, rep_))
      throwEntryError("int64", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vInt32

  std::vector<long long> Entry::getVInt64() const {
    if (type_ != kTvint64)
      throwValueError("vector<int64>");
    std::vector<long long> val;
    if (!decode(val, rep_))
      throwEntryError("vector<int64>", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // Uint32

  unsigned Entry::getUInt32() const {
    if (type_ != kTuint32)
      throwValueError("unsigned int");
    unsigned val;
    if (!decode(val, rep_))
      throwEntryError("unsigned int", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vUint32

  std::vector<unsigned> Entry::getVUInt32() const {
    if (type_ != kTvuint32)
      throwValueError("vector<unsigned int>");
    std::vector<unsigned> val;
    if (!decode(val, rep_))
      throwEntryError("vector<unsigned int>", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // Uint64

  unsigned long long Entry::getUInt64() const {
    if (type_ != kTuint64)
      throwValueError("uint64");
    unsigned long long val;
    if (!decode(val, rep_))
      throwEntryError("uint64", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vUint64

  std::vector<unsigned long long> Entry::getVUInt64() const {
    if (type_ != kTvuint64)
      throwValueError("vector<uint64>");
    std::vector<unsigned long long> val;
    if (!decode(val, rep_))
      throwEntryError("vector<uint64>", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // Double

  double Entry::getDouble() const {
    if (type_ != kTdouble)
      throwValueError("double");
    double val;
    if (!decode(val, rep_))
      throwEntryError("double", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vDouble

  std::vector<double> Entry::getVDouble() const {
    if (type_ != kTvdouble)
      throwValueError("vector<double>");
    std::vector<double> val;
    if (!decode(val, rep_))
      throwEntryError("vector<double>", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // String

  std::string Entry::getString() const {
    if (type_ != kTstringRaw and type_ != kTstringHex)
      throwValueError("string");
    std::string val;
    if (type_ == kTstringHex) {
      if (!decode_deprecated(val, rep_)) {
        throwEntryError("string", rep_);
      }
    } else if (!decode(val, rep_)) {
      throwEntryError("string", rep_);
    }
    return val;
  }

  // ----------------------------------------------------------------------
  // vString

  std::vector<std::string> Entry::getVString() const {
    if (type_ != kTvstringRaw and type_ != kTvstringHex)
      throwValueError("vector<string>");
    std::vector<std::string> val;
    if (type_ == kTvstringHex) {
      if (!decode_deprecated(val, rep_))
        throwEntryError("vector<string>", rep_);
    } else if (!decode(val, rep_)) {
      throwEntryError("vector<string>", rep_);
    }
    return val;
  }

  // ----------------------------------------------------------------------
  // FileInPath

  FileInPath Entry::getFileInPath() const {
    if (type_ != kTFileInPath)
      throwValueError("FileInPath");
    FileInPath val;
    if (!decode(val, rep_))
      throwEntryError("FileInPath", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // InputTag

  InputTag Entry::getInputTag() const {
    if (type_ != kTInputTag)
      throwValueError("InputTag");
    InputTag val;
    if (!decode(val, rep_))
      throwEntryError("InputTag", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VInputTag

  std::vector<InputTag> Entry::getVInputTag() const {
    if (type_ != kTVInputTag)
      throwValueError("VInputTag");
    std::vector<InputTag> val;
    if (!decode(val, rep_))
      throwEntryError("VInputTag", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // ESInputTag

  ESInputTag Entry::getESInputTag() const {
    if (type_ != kTESInputTag)
      throwValueError("ESInputTag");
    ESInputTag val;
    if (!decode(val, rep_))
      throwEntryError("ESInputTag", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VESInputTag

  std::vector<ESInputTag> Entry::getVESInputTag() const {
    if (type_ != kTVESInputTag)
      throwValueError("VESInputTag");
    std::vector<ESInputTag> val;
    if (!decode(val, rep_))
      throwEntryError("VESInputTag", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // EventID

  EventID Entry::getEventID() const {
    if (type_ != kTEventID)
      throwValueError("EventID");
    EventID val;
    if (!decode(val, rep_))
      throwEntryError("EventID", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VEventID

  std::vector<EventID> Entry::getVEventID() const {
    if (type_ != kTVEventID)
      throwValueError("VEventID");
    std::vector<EventID> val;
    if (!decode(val, rep_))
      throwEntryError("EventID", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  LuminosityBlockID Entry::getLuminosityBlockID() const {
    if (type_ != kTLuminosityBlockID)
      throwValueError("LuminosityBlockID");
    LuminosityBlockID val;
    if (!decode(val, rep_))
      throwEntryError("LuminosityBlockID", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  std::vector<LuminosityBlockID> Entry::getVLuminosityBlockID() const {
    if (type_ != kTVLuminosityBlockID)
      throwValueError("VLuminosityBlockID");
    std::vector<LuminosityBlockID> val;
    if (!decode(val, rep_))
      throwEntryError("LuminosityBlockID", rep_);
    return val;
  }

  // LuminosityBlockRange

  LuminosityBlockRange Entry::getLuminosityBlockRange() const {
    if (type_ != kTLuminosityBlockRange)
      throwValueError("LuminosityBlockRange");
    LuminosityBlockRange val;
    if (!decode(val, rep_))
      throwEntryError("LuminosityBlockRange", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  std::vector<LuminosityBlockRange> Entry::getVLuminosityBlockRange() const {
    if (type_ != kTVLuminosityBlockRange)
      throwValueError("VLuminosityBlockRange");
    std::vector<LuminosityBlockRange> val;
    if (!decode(val, rep_))
      throwEntryError("LuminosityBlockRange", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // EventRange

  EventRange Entry::getEventRange() const {
    if (type_ != kTEventRange)
      throwValueError("EventRange");
    EventRange val;
    if (!decode(val, rep_))
      throwEntryError("EventRange", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // VEventRange

  std::vector<EventRange> Entry::getVEventRange() const {
    if (type_ != kTVEventRange)
      throwValueError("VEventRange");
    std::vector<EventRange> val;
    if (!decode(val, rep_))
      throwEntryError("EventRange", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // ParameterSet

  ParameterSet Entry::getPSet() const {
    if (type_ != kTPSet)
      throwValueError("ParameterSet");
    ParameterSet val;
    if (!decode(val, rep_))
      throwEntryError("ParameterSet", rep_);
    return val;
  }

  // ----------------------------------------------------------------------
  // vPSet

  std::vector<ParameterSet> Entry::getVPSet() const {
    if (type_ != kTvPSet)
      throwValueError("vector<ParameterSet>");
    std::vector<ParameterSet> val;
    if (!decode(val, rep_))
      throwEntryError("vector<ParameterSet>", rep_);
    return val;
  }

  std::ostream& operator<<(std::ostream& os, Entry const& entry) {
    os << typeFromCode(entry.typeCode()) << " " << (entry.isTracked() ? "tracked " : "untracked ") << " = ";

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
      case kTstringRaw: {
        os << "'" << entry.getString() << "'";
        break;
      }
      case kTvstringRaw: {
        os << "{";
        std::string_view start = "'";
        std::string_view const between(",'");
        std::vector<std::string> strings = entry.getVString();
        for (auto const& s : strings) {
          os << start << s << "'";
          start = between;
        }
        os << "}";
        break;
      }
      case kTstringHex: {
        os << "'" << entry.getString() << "'";
        break;
      }
      case kTvstringHex: {
        os << "{";
        std::string_view start = "'";
        std::string_view const between(",'");
        std::vector<std::string> strings = entry.getVString();
        for (auto const& s : strings) {
          os << start << s << "'";
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
        os << entry.rep_;
        break;
      }
    }

    return os;
  }

  // Helper functions for throwing exceptions

  void Entry::throwValueError(char const* expectedType) const {
    throw Exception(errors::Configuration, "ValueError")
        << "type of " << name_ << " is expected to be " << expectedType << " but declared as " << typeFromCode(type_);
  }

  void Entry::throwEntryError(char const* expectedType, std::string const& badRep) const {
    throw Exception(errors::Configuration, "EntryError") << "can not convert representation of " << name_ << ": "
                                                         << badRep << " to value of type " << expectedType << " ";
  }

  void Entry::throwEncodeError(char const* type) const {
    throw Exception(errors::Configuration, "EncodingError") << "can not encode " << name_ << " as type: " << type;
  }

}  // namespace edm
