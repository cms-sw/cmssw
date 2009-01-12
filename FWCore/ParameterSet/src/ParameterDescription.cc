// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescription
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:35:43 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <boost/cstdint.hpp>
#include <cassert>

#define TYPE_TO_ENUM(type,e_val) template<> ParameterTypes ParameterTypeToEnum::toEnum<type >(){ return e_val; }
#define TYPE_TO_NAME(type) case k_ ## type: return #type

namespace edm {

  class EventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class FileInPath;

  TYPE_TO_ENUM(int,k_int32)
  TYPE_TO_ENUM(std::vector<int>,k_vint32)
  TYPE_TO_ENUM(unsigned,k_uint32)
  TYPE_TO_ENUM(std::vector<unsigned>,k_vuint32)
  TYPE_TO_ENUM(boost::int64_t,k_int64)
  TYPE_TO_ENUM(std::vector<boost::int64_t>,k_vint64)
  TYPE_TO_ENUM(boost::uint64_t,k_uint64)
  TYPE_TO_ENUM(std::vector<boost::uint64_t>,k_vuint64)
  TYPE_TO_ENUM(double,k_double)
  TYPE_TO_ENUM(std::vector<double>,k_vdouble)
  TYPE_TO_ENUM(bool,k_bool)
  TYPE_TO_ENUM(std::string,k_string)
  TYPE_TO_ENUM(std::vector<std::string>,k_vstring)
  TYPE_TO_ENUM(edm::EventID,k_EventID)
  TYPE_TO_ENUM(std::vector<edm::EventID>,k_VEventID)
  TYPE_TO_ENUM(edm::LuminosityBlockID,k_LuminosityBlockID)
  TYPE_TO_ENUM(std::vector<edm::LuminosityBlockID>,k_VLuminosityBlockID)
  TYPE_TO_ENUM(edm::InputTag,k_InputTag)
  TYPE_TO_ENUM(std::vector<edm::InputTag>,k_VInputTag)
  TYPE_TO_ENUM(edm::FileInPath,k_FileInPath)
  TYPE_TO_ENUM(edm::LuminosityBlockRange,k_LuminosityBlockRange)
  TYPE_TO_ENUM(std::vector<edm::LuminosityBlockRange>,k_VLuminosityBlockRange)
  TYPE_TO_ENUM(edm::EventRange,k_EventRange)
  TYPE_TO_ENUM(std::vector<edm::EventRange>,k_VEventRange)
  // These are intentionally not implemented to prevent one
  // from calling add<ParameterSet>.  One should call
  // add<ParameterSetDescription> instead.
  // TYPE_TO_ENUM(edm::ParameterSet,k_PSet)
  // TYPE_TO_ENUM(std::vector<edm::ParameterSet>,k_VPSet)

  std::string parameterTypeEnumToString(ParameterTypes iType) {
    switch(iType) {
      TYPE_TO_NAME(int32);
      TYPE_TO_NAME(vint32);
      TYPE_TO_NAME(uint32);
      TYPE_TO_NAME(vuint32);
      TYPE_TO_NAME(int64);
      TYPE_TO_NAME(vint64);
      TYPE_TO_NAME(uint64);
      TYPE_TO_NAME(vuint64);
      TYPE_TO_NAME(double);
      TYPE_TO_NAME(vdouble);
      TYPE_TO_NAME(bool);
      TYPE_TO_NAME(string);
      TYPE_TO_NAME(vstring);
      TYPE_TO_NAME(EventID);
      TYPE_TO_NAME(VEventID);
      TYPE_TO_NAME(LuminosityBlockID);
      TYPE_TO_NAME(VLuminosityBlockID);
      TYPE_TO_NAME(InputTag);
      TYPE_TO_NAME(VInputTag);
      TYPE_TO_NAME(FileInPath);
      TYPE_TO_NAME(PSet);
      TYPE_TO_NAME(VPSet);
      TYPE_TO_NAME(LuminosityBlockRange);
      TYPE_TO_NAME(VLuminosityBlockRange);
      TYPE_TO_NAME(EventRange);
      TYPE_TO_NAME(VEventRange);
    default:
      assert(false);
    }
    return "";
  }

  ParameterDescription::ParameterDescription(std::string const& iLabel,
                                             ParameterTypes iType,
                                             bool isTracked,
                                             bool isOptional)
    :label_(iLabel),
     type_(iType),
     isTracked_(isTracked),
     isOptional_(isOptional)
  { }

  ParameterDescription::ParameterDescription(char const* iLabel,
                                             ParameterTypes iType,
                                             bool isTracked,
                                             bool isOptional)
    :label_(iLabel),
     type_(iType),
     isTracked_(isTracked),
     isOptional_(isOptional)
  { }

  ParameterDescription::~ParameterDescription() { }

  void
  ParameterDescription::throwParameterNotDefined() const {
    std::string tr("an untracked");
    if (isTracked()) tr = "a tracked";

    throw edm::Exception(errors::Configuration)
      << "Required parameter \"" << label() << "\" not defined "
      "as " << tr << " " << parameterTypeEnumToString(type()) << ".";
  }

  void
  ParameterDescription::writeCfi(std::ostream & os, int indentation) const {

    os << label()
       << " = cms.";
    if (!isTracked()) os << "untracked.";
    os << parameterTypeEnumToString(type())
       << "(";
    writeCfi_(os, indentation);
    os << ")";
  }
}
