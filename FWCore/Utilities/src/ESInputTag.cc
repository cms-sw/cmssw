// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ESInputTag
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 13:01:56 CST 2009
//

// system include files
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace edm;

ESInputTag::ESInputTag() = default;

ESInputTag::ESInputTag(const std::string& moduleLabel, const std::string& dataLabel):
module_(moduleLabel),
data_(dataLabel)
{
}

ESInputTag::ESInputTag(const std::string& iEncodedValue)
{
   // string is delimited by colons
   std::vector<std::string> tokens = tokenize(iEncodedValue, ":");
   int nwords = tokens.size();
   if(nwords > 2) {
      throw edm::Exception(errors::Configuration,"ESInputTag")
      << "ESInputTag " << iEncodedValue << " has " << nwords << " tokens but only up two 2 are allowed.";
   }
   if(nwords > 0) module_ = tokens[0];
   if(nwords > 1) data_ = tokens[1];
}

//
// const member functions
//
bool
ESInputTag::operator==(const edm::ESInputTag& iRHS) const
{
  return module_ == iRHS.module_ &&
    data_ == iRHS.data_;
}

std::string ESInputTag::encode() const {
   static std::string const separator(":");
   std::string result = module_;
   if(!data_.empty()) {
      result += separator + data_;
   }
   return result;
}

namespace edm {
  std::ostream&
  operator<<(std::ostream& os, ESInputTag const& tag)
  {
    os << "Module label: " << tag.module()
       << " Data label: " << tag.data();
    return os;
  }
}
