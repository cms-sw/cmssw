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
// $Id: ParameterDescription.cc,v 1.3 2008/11/14 19:41:23 wdd Exp $
//

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <boost/cstdint.hpp>

#define TYPE_TO_ENUM(type,e_val) template<> ParameterTypes ParameterTypeToEnum::toEnum<type >(){ return e_val; }
#define TYPE_TO_NAME(type) case k_ ## type: return #type

namespace edm {

  class EventID;
  class LuminosityBlockID;
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
  TYPE_TO_ENUM(std::vector<bool>,k_vbool)
  TYPE_TO_ENUM(std::string,k_string)
  TYPE_TO_ENUM(std::vector<std::string>,k_vstring)
  TYPE_TO_ENUM(edm::EventID,k_EventID)
  TYPE_TO_ENUM(std::vector<edm::EventID>,k_VEventID)
  TYPE_TO_ENUM(edm::LuminosityBlockID,k_LuminosityBlockID)
  TYPE_TO_ENUM(std::vector<edm::LuminosityBlockID>,k_VLuminosityBlockID)
  TYPE_TO_ENUM(edm::InputTag,k_InputTag)
  TYPE_TO_ENUM(std::vector<edm::InputTag>,k_VInputTag)
  TYPE_TO_ENUM(edm::FileInPath,k_FileInPath)
  TYPE_TO_ENUM(edm::ParameterSet,k_PSet)
  TYPE_TO_ENUM(std::vector<edm::ParameterSet>,k_VPSet)

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
      TYPE_TO_NAME(vbool);
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
    default:
      assert(false);
    }
    return "";
  }

  ParameterDescription::ParameterDescription(const std::string& iLabel,
                                             bool isTracked,
                                             bool optional,
                                             ParameterTypes iType)
  :label_(iLabel),
   type_(iType),
   isTracked_(isTracked),
   optional_(optional),
   parameterSetDescription_(),
   parameterSetDescriptions_()
  {
    if (type() == k_PSet) {
      parameterSetDescription_.reset(new ParameterSetDescription);
    }
    else if (type() == k_VPSet) {
      parameterSetDescriptions_.reset(new std::vector<ParameterSetDescription>);
    }
  }

  ParameterDescription::~ParameterDescription() { }

  void
  ParameterDescription::validate(const ParameterSet& pset) const {

    bool exists;
    validate_(pset, exists);

    if (exists && type() == k_PSet) {
      validateParameterSetDescription(pset);
    }
    else if (exists && type() == k_VPSet) {
      validateParameterSetDescriptions(pset);
    }
  }

  void
  ParameterDescription::validateParameterSetDescription(const ParameterSet& pset) const {
    ParameterSet containedPSet;
    if (isTracked()) {
      containedPSet = pset.getParameter<ParameterSet>(label());
    }
    else {
      containedPSet = pset.getUntrackedParameter<ParameterSet>(label());
    }
    parameterSetDescription_->validate(containedPSet);
  }

  void
  ParameterDescription::validateParameterSetDescriptions(const ParameterSet& pset) const {

    std::vector<ParameterSet> containedPSets;
    if (isTracked()) {
      containedPSets = pset.getParameter<std::vector<ParameterSet> >(label());
    }
    else {
      containedPSets = pset.getUntrackedParameter<std::vector<ParameterSet> >(label());
    }
    if (containedPSets.size() != parameterSetDescriptions_->size()) {
      throw edm::Exception(errors::Configuration)
        << "Unexpected number of ParameterSets in vector of parameter sets named \"" << label() << "\".";
    }
    int i = 0;
    for (std::vector<ParameterSetDescription>::const_iterator iter = parameterSetDescriptions_->begin(),
	                                                      iEnd = parameterSetDescriptions_->end();
         iter != iEnd;
         ++iter, ++i) {
      iter->validate(containedPSets[i]);
    }
  }

  void
  ParameterDescription::throwParameterNotDefined() const {
    throw edm::Exception(errors::Configuration)
      << "Parameter \"" << label() << "\" not defined.";
  }
}
