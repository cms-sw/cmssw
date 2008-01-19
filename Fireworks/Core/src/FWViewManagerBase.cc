// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewManagerBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sat Jan  5 10:56:17 EST 2008
// $Id: FWViewManagerBase.cc,v 1.2 2008/01/11 00:49:03 chrjones Exp $
//

// system include files
#include "TClass.h"
#include "TROOT.h"
#include <string>
#include <iostream>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewManagerBase::FWViewManagerBase(const char* iPostfix):
  m_detIdToGeo(0),
  m_builderNamePostfixes(&iPostfix, &iPostfix+1)
{
}

// FWViewManagerBase::FWViewManagerBase(const FWViewManagerBase& rhs)
// {
//    // do actual copying here;
// }

FWViewManagerBase::~FWViewManagerBase()
{
}

//
// assignment operators
//
// const FWViewManagerBase& FWViewManagerBase::operator=(const FWViewManagerBase& rhs)
// {
//   //An exception safe implementation is
//   FWViewManagerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void* 
FWViewManagerBase::createInstanceOf(const TClass* iBaseClass,
				    const char* iNameOfClass)
{
   //create proxy builders
   Int_t error;
   assert(iBaseClass !=0);

   //does the class already exist?
   TClass *c = TClass::GetClass( iNameOfClass );
   if(0==c) {
      //try to load a macro of that name
      
      //How can I tell if this succeeds or failes? error and value are always 0!
      // I could not make the non-compiled mechanism to work without seg-faults
      Int_t value = gROOT->LoadMacro( (std::string(iNameOfClass)+".C+").c_str(), &error );
      c = TClass::GetClass( iNameOfClass );
      if(0==c ) {
	 std::cerr <<"failed to find "<< iNameOfClass << std::endl;
	 return 0;
      }
   }
   void* inst = c->New();
   void* baseClassInst = c->DynamicCast(iBaseClass,inst);
   if(0==baseClassInst) {
     std::cerr<<"conversion to "<<iBaseClass->ClassName() << " failed"<<std::endl;
      return 0;
   }
   return baseClassInst;
}

//
// const member functions
//
bool
FWViewManagerBase::useableBuilder(const std::string& iName) const
{
  for( std::vector<std::string>::const_iterator itPostfix = m_builderNamePostfixes.begin();
      itPostfix != m_builderNamePostfixes.end();
      ++itPostfix) {
    if(std::string::npos != iName.find( *itPostfix) ) {
      return true;
    }
  }
  return false;
}

//
// static member functions
//
