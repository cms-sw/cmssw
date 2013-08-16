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
#include <vector>

// user include files
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ESInputTag::ESInputTag()
{
}

ESInputTag::ESInputTag(const std::string& moduleLabel, const std::string& dataLabel):
module_(moduleLabel),
data_(dataLabel)
{
}

ESInputTag::ESInputTag( const std::string& iEncodedValue)
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
// ESInputTag::ESInputTag(const ESInputTag& rhs)
// {
//    // do actual copying here;
// }

//ESInputTag::~ESInputTag()
//{
//}

//
// assignment operators
//
// const ESInputTag& ESInputTag::operator=(const ESInputTag& rhs)
// {
//   //An exception safe implementation is
//   ESInputTag temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

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

//
// static member functions
//
