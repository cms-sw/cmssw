// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterSetterBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:16:20 EST 2008
// $Id: FWParameterSetterBase.cc,v 1.1 2008/03/11 02:43:55 chrjones Exp $
//

// system include files
#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "TGedFrame.h"

#include <assert.h>
#include <iostream>

// user include files
#include "FWCore/Utilities/interface/TypeID.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWParameterSetterBase::FWParameterSetterBase():
m_frame(0)
{
}

// FWParameterSetterBase::FWParameterSetterBase(const FWParameterSetterBase& rhs)
// {
//    // do actual copying here;
// }

FWParameterSetterBase::~FWParameterSetterBase()
{
}

//
// assignment operators
//
// const FWParameterSetterBase& FWParameterSetterBase::operator=(const FWParameterSetterBase& rhs)
// {
//   //An exception safe implementation is
//   FWParameterSetterBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void 
FWParameterSetterBase::attach(FWParameterBase* iBase, TGedFrame* iFrame)
{
   m_frame=iFrame;
   attach(iBase);
}


//
// const member functions
//
void 
FWParameterSetterBase::update() const
{
   m_frame->Update();
}

//
// static member functions
//
FWParameterSetterBase* 
FWParameterSetterBase::makeSetterFor(FWParameterBase* iParam)
{
   static std::map<edm::TypeID,ROOT::Reflex::Type> s_paramToSetterMap;
   edm::TypeID paramType( typeid(*iParam) );
   std::map<edm::TypeID,ROOT::Reflex::Type>::iterator itFind = s_paramToSetterMap.find(paramType);
   if( itFind == s_paramToSetterMap.end() ) {
      ROOT::Reflex::Type paramClass( ROOT::Reflex::Type::ByTypeInfo(typeid(*iParam)) );
      if(paramClass == ROOT::Reflex::Type() ) {
         std::cout << "PROGRAMMING ERROR: the type "<<typeid(*iParam).name()<< " is not known to REFLEX" <<std::endl;
      }
      assert(paramClass != ROOT::Reflex::Type() );
      
      //the corresponding setter has the same name but with 'Setter' at the end
      std::string name = paramClass.Name(ROOT::Reflex::SCOPED);
      name += "Setter";
      
      ROOT::Reflex::Type setterClass( ROOT::Reflex::Type::ByName( name ) );
      if(setterClass == ROOT::Reflex::Type() ) {
         std::cout << "PROGRAMMING ERROR: the type "<<name<< " is not known to REFLEX" <<std::endl;
      }
      assert(setterClass != ROOT::Reflex::Type());
      
      s_paramToSetterMap[paramType]=setterClass;
      itFind = s_paramToSetterMap.find(paramType);
   }
   //create the instance we want
   ROOT::Reflex::Object setterObj = itFind->second.Construct();
   
   //make it into the base class
   ROOT::Reflex::Type s_setterBaseType( ROOT::Reflex::Type::ByTypeInfo( typeid(FWParameterSetterBase) ) );
   assert(s_setterBaseType != ROOT::Reflex::Type());
   setterObj = setterObj.CastObject(s_setterBaseType);
   
   return reinterpret_cast<FWParameterSetterBase*>( setterObj.Address() );
}
