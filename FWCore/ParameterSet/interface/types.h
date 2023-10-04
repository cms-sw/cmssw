#ifndef FWCore_ParameterSet_types_h
#define FWCore_ParameterSet_types_h

// ----------------------------------------------------------------------
// declaration of type encoding/decoding functions
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prolog

// ----------------------------------------------------------------------
// prerequisite source files and headers

#include <string>
#include <string_view>
#include <vector>
#include <optional>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// ----------------------------------------------------------------------
// contents

namespace edm {
  //            destination    source

  // Bool
  bool decode(bool&, std::string_view);
  bool encode(std::string&, bool);

  // vBool
  bool decode(std::vector<bool>&, std::string_view);
  bool encode(std::string&, std::vector<bool> const&);

  // Int32
  bool decode(int&, std::string_view);
  bool encode(std::string&, int);

  // vInt32
  bool decode(std::vector<int>&, std::string_view);
  bool encode(std::string&, std::vector<int> const&);

  // Uint32
  bool decode(unsigned int&, std::string_view);
  bool encode(std::string&, unsigned int);

  // vUint32
  bool decode(std::vector<unsigned int>&, std::string_view);
  bool encode(std::string&, std::vector<unsigned int> const&);

  // Int64
  bool decode(long long&, std::string_view);
  bool encode(std::string&, long long);

  // vInt64
  bool decode(std::vector<long long>&, std::string_view);
  bool encode(std::string&, std::vector<long long> const&);

  // Uint64
  bool decode(unsigned long long&, std::string_view);
  bool encode(std::string&, unsigned long long);

  // vUint64
  bool decode(std::vector<unsigned long long>&, std::string_view);
  bool encode(std::string&, std::vector<unsigned long long> const&);

  // Double
  bool decode(double&, std::string_view);
  bool encode(std::string&, double);

  // vDouble
  bool decode(std::vector<double>&, std::string_view);
  bool encode(std::string&, std::vector<double> const&);

  // String
  bool decode(std::string&, std::string_view);
  bool encode(std::string&, std::string const&);
  std::optional<std::string_view> decode_string_extent(std::string_view from);

  // vString
  bool decode(std::vector<std::string>&, std::string_view);
  bool encode(std::string&, std::vector<std::string> const&);
  bool decode_element(std::string&, std::string_view);
  bool encode_element(std::string&, std::string const&);
  std::optional<std::string_view> decode_vstring_extent(std::string_view from);

  // String old, kept for backwards compatibility
  bool decode_deprecated(std::string&, std::string_view);
  bool encode_deprecated(std::string&, std::string const&);

  // vString old, kept for backwards compatibility
  bool decode_deprecated(std::vector<std::string>&, std::string_view);
  bool encode_deprecated(std::string&, std::vector<std::string> const&);

  // FileInPath
  bool decode(edm::FileInPath&, std::string_view);
  bool encode(std::string&, edm::FileInPath const&);

  // InputTag
  bool decode(edm::InputTag&, std::string_view);
  bool encode(std::string&, edm::InputTag const&);

  // VInputTag
  bool decode(std::vector<edm::InputTag>&, std::string_view);
  bool encode(std::string&, std::vector<edm::InputTag> const&);

  // ESInputTag
  bool decode(edm::ESInputTag&, std::string_view);
  bool encode(std::string&, edm::ESInputTag const&);

  // VESInputTag
  bool decode(std::vector<edm::ESInputTag>&, std::string_view);
  bool encode(std::string&, std::vector<edm::ESInputTag> const&);

  // EventID
  bool decode(edm::EventID&, std::string_view);
  bool encode(std::string&, edm::EventID const&);

  // VEventID
  bool decode(std::vector<edm::EventID>&, std::string_view);
  bool encode(std::string&, std::vector<edm::EventID> const&);

  // LuminosityBlockID
  bool decode(edm::LuminosityBlockID&, std::string_view);
  bool encode(std::string&, edm::LuminosityBlockID const&);

  // VLuminosityBlockID
  bool decode(std::vector<edm::LuminosityBlockID>&, std::string_view);
  bool encode(std::string&, std::vector<edm::LuminosityBlockID> const&);

  // LuminosityBlockRange
  bool decode(edm::LuminosityBlockRange&, std::string_view);
  bool encode(std::string&, edm::LuminosityBlockRange const&);

  // VLuminosityBlockRange
  bool decode(std::vector<edm::LuminosityBlockRange>&, std::string_view);
  bool encode(std::string&, std::vector<edm::LuminosityBlockRange> const&);

  // EventRange
  bool decode(edm::EventRange&, std::string_view);
  bool encode(std::string&, edm::EventRange const&);

  // VEventRange
  bool decode(std::vector<edm::EventRange>&, std::string_view);
  bool encode(std::string&, std::vector<edm::EventRange> const&);

  // ParameterSet
  bool decode(ParameterSet&, std::string_view);
  bool encode(std::string&, ParameterSet const&);
  std::optional<std::string_view> decode_pset_extent(std::string_view from);

  // vPSet
  bool decode(std::vector<ParameterSet>&, std::string_view);
  bool encode(std::string&, std::vector<ParameterSet> const&);
  std::optional<std::string_view> decode_vpset_extent(std::string_view from);

}  // namespace edm

// ----------------------------------------------------------------------
// epilog

#endif
