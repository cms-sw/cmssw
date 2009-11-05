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
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// ----------------------------------------------------------------------
// contents

namespace edm
{
  //            destination    source

  // Bool
  bool  decode(bool        &, std::string const&);
  bool  encode(std::string &, bool);

  // vBool
  bool  decode(std::vector<bool> &, std::string       const&);
  bool  encode(std::string       &, std::vector<bool> const&);

  // Int32
  bool  decode(int         &, std::string const&);
  bool  encode(std::string &, int);

  // vInt32
  bool  decode(std::vector<int> &, std::string      const&);
  bool  encode(std::string      &, std::vector<int> const&);

  // Uint32
  bool  decode(unsigned    &, std::string const&);
  bool  encode(std::string &, unsigned);

  // vUint32
  bool  decode(std::vector<unsigned> &, std::string           const&);
  bool  encode(std::string           &, std::vector<unsigned> const&);

  // Int64
  bool  decode(long long     &, std::string const&);
  bool  encode(std::string &, long long);

  // vInt64
  bool  decode(std::vector<long long> &, std::string      const&);
  bool  encode(std::string      &, std::vector<long long> const&);

  // Uint64
  bool  decode(unsigned long long    &, std::string const&);
  bool  encode(std::string &, unsigned long long);

  // vUint64
  bool  decode(std::vector<unsigned long long> &, std::string           const&);
  bool  encode(std::string           &, std::vector<unsigned long long> const&);

  // Double
  bool  decode(double      &, std::string const&);
  bool  encode(std::string &, double);

  // vDouble
  bool  decode(std::vector<double> &, std::string         const&);
  bool  encode(std::string         &, std::vector<double> const&);

  // String
  bool  decode(std::string &, std::string const&);
  bool  encode(std::string &, std::string const&);

  // vString
  bool  decode(std::vector<std::string> &, std::string              const&);
  bool  encode(std::string              &, std::vector<std::string> const&);

  // FileInPath
  bool  decode(edm::FileInPath &, std::string const&);
  bool  encode(std::string &, edm::FileInPath const&);

  // InputTag
  bool  decode(edm::InputTag&, std::string const&);
  bool  encode(std::string &, edm::InputTag const&);

  // VInputTag
  bool  decode(std::vector<edm::InputTag>&, std::string const&);
  bool  encode(std::string &, std::vector<edm::InputTag> const&);

   // ESInputTag
   bool  decode(edm::ESInputTag&, std::string const&);
   bool  encode(std::string &, edm::ESInputTag const&);
   
   // VESInputTag
   bool  decode(std::vector<edm::ESInputTag>&, std::string const&);
   bool  encode(std::string &, std::vector<edm::ESInputTag> const&);
   
   // EventID
  bool  decode(edm::MinimalEventID&, std::string const&);
  bool  encode(std::string &, edm::MinimalEventID const&);

  // VEventID
  bool  decode(std::vector<edm::MinimalEventID>&, std::string const&);
  bool  encode(std::string &, std::vector<edm::MinimalEventID> const&);

  // LuminosityBlockID
  bool  decode(edm::LuminosityBlockID&, std::string const&);
  bool  encode(std::string &, edm::LuminosityBlockID const&);

  // VLuminosityBlockID
  bool  decode(std::vector<edm::LuminosityBlockID>&, std::string const&);
  bool  encode(std::string &, std::vector<edm::LuminosityBlockID> const&);

  // LuminosityBlockRange
  bool  decode(edm::LuminosityBlockRange&, std::string const&);
  bool  encode(std::string &, edm::LuminosityBlockRange const&);

  // VLuminosityBlockRange
  bool  decode(std::vector<edm::LuminosityBlockRange>&, std::string const&);
  bool  encode(std::string &, std::vector<edm::LuminosityBlockRange> const&);

  // EventRange
  bool  decode(edm::EventRange&, std::string const&);
  bool  encode(std::string &, edm::EventRange const&);

  // VEventRange
  bool  decode(std::vector<edm::EventRange>&, std::string const&);
  bool  encode(std::string &, std::vector<edm::EventRange> const&);

  // ParameterSet
  bool  decode(ParameterSet &, std::string  const&);
  bool  encode(std::string  &, ParameterSet const&);

  // vPSet
  bool  decode(std::vector<ParameterSet> &, std::string               const&);
  bool  encode(std::string               &, std::vector<ParameterSet> const&);

}  // namespace edm


// ----------------------------------------------------------------------
// epilog

#endif
