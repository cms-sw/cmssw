// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataKeyTags
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:25:33 EST 2005
// $Id: DataKeyTags.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $
//

// system include files
#include <cstring>

// user include files
#include "FWCore/Framework/interface/DataKeyTags.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor

//
// assignment operators
//
// const DataKeyTags& DataKeyTags::operator=(const DataKeyTags& rhs)
// {
//   //An exception safe implementation is
//   DataKeyTags temp(rhs);
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
SimpleStringTag::operator==(const SimpleStringTag& iRHS) const
{
   return (0 == std::strcmp(tag_, iRHS.tag_));
}

bool
SimpleStringTag::operator<(const SimpleStringTag& iRHS) const
{
   return (0 > std::strcmp(tag_, iRHS.tag_));
}

//
// static member functions
//
   }
}
