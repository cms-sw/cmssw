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
// $Id: FWEDProductRepresentationChecker.cc,v 1.2 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include "TClass.h"

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
FWEDProductRepresentationChecker::FWEDProductRepresentationChecker(const std::string& iTypeidName,
                                                                   const std::string& iPurpose) :
   FWRepresentationCheckerBase(iPurpose),
   m_typeidName(iTypeidName)
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
   TClass* clss = TClass::GetClass(iTypeName.c_str());
   if(0==clss || clss->GetTypeInfo()==0) {
      return FWRepresentationInfo();
   }
   if(clss->GetTypeInfo()->name() == m_typeidName) {
      return FWRepresentationInfo(purpose(),0);
   }
   return FWRepresentationInfo();
}

//
// static member functions
//
