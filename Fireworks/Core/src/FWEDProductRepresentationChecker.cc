// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEDProductRepresentationChecker
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 15:20:10 EST 2008
// $Id: FWEDProductRepresentationChecker.cc,v 1.1 2008/11/14 16:29:31 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"

#include "Fireworks/Core/interface/FWRepresentationInfo.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEDProductRepresentationChecker::FWEDProductRepresentationChecker(const std::string& iTypeName,
                                                                   const std::string& iPurpose) :
   FWRepresentationCheckerBase(iPurpose),
   m_typeName(iTypeName)
{
}

// FWEDProductRepresentationChecker::FWEDProductRepresentationChecker(const FWEDProductRepresentationChecker& rhs)
// {
//    // do actual copying here;
// }

//FWEDProductRepresentationChecker::~FWEDProductRepresentationChecker()
//{
//}

//
// assignment operators
//
// const FWEDProductRepresentationChecker& FWEDProductRepresentationChecker::operator=(const FWEDProductRepresentationChecker& rhs)
// {
//   //An exception safe implementation is
//   FWEDProductRepresentationChecker temp(rhs);
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
FWRepresentationInfo
FWEDProductRepresentationChecker::infoFor(const std::string& iTypeName) const
{
   if(iTypeName == m_typeName) {
      return FWRepresentationInfo(purpose(),0);
   }
   return FWRepresentationInfo();
}

//
// static member functions
//
