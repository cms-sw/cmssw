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
// $Id: FWParameterSetterBase.cc,v 1.17 2013/02/18 23:42:56 wmtan Exp $
//

// system include files
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include <assert.h>
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "FWCore/Utilities/interface/TypeID.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"
#include "Fireworks/Core/interface/fwLog.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWParameterSetterBase::FWParameterSetterBase() :
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
FWParameterSetterBase::attach(FWParameterBase* iBase, FWParameterSetterEditorBase* iFrame)
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
   if (m_frame != 0)
      m_frame->updateEditor();
}

//
// static member functions
//

boost::shared_ptr<FWParameterSetterBase>
FWParameterSetterBase::makeSetterFor(FWParameterBase* iParam)
{
   static std::map<edm::TypeID,edm::TypeWithDict> s_paramToSetterMap;
   edm::TypeID paramType( typeid(*iParam) );
   std::map<edm::TypeID,edm::TypeWithDict>::iterator itFind = s_paramToSetterMap.find(paramType);
   if (itFind == s_paramToSetterMap.end())
   {
      edm::TypeWithDict paramClass(typeid(*iParam));
      if (paramClass == edm::TypeWithDict())
      {
         fwLog(fwlog::kError) << " the type "<<typeid(*iParam).name()<< " is not known to Root" <<std::endl;
      }
      assert(paramClass != edm::TypeWithDict() );

      //the corresponding setter has the same name but with 'Setter' at the end
      std::string name = paramClass.name();
      // FIXME: there was a convention between parameter class names and associated
      //        setters. The following works around the problem introduced by
      //        the generic parameter class but it is clear that a better 
      //        way of doing the binding is required. Notice that there are only 5 
      //        different type of FW*Parameter.
      if (name == "FWGenericParameter<bool>")
         name = "FWBoolParameterSetter";
      else if (name == "FWGenericParameter<std::string>")
         name = "FWStringParameterSetter";
      else if (name == "FWGenericParameter<std::basic_string<char> >")
         name = "FWStringParameterSetter"; 
      else if (name == "FWGenericParameterWithRange<double>")
         name = "FWDoubleParameterSetter";
      else if (name == "FWGenericParameterWithRange<long int>")
         name = "FWLongParameterSetter";
      else if (name == "FWGenericParameterWithRange<long>")
         name = "FWLongParameterSetter";
      else
         name += "Setter";

      edm::TypeWithDict setterClass( edm::TypeWithDict::byName( name ) );
      if (setterClass == edm::TypeWithDict())
      {
         fwLog(fwlog::kError) << " the type "<<name<< " has no dictionary" <<std::endl;
      }
      assert(setterClass != edm::TypeWithDict());

      s_paramToSetterMap[paramType]=setterClass;
      itFind = s_paramToSetterMap.find(paramType);
   }
   //create the instance we want
   //NOTE: 'construct' will use 'malloc' to allocate the memory if the object is not of class type.
   // This means the object cannot be deleted using 'delete'!  So we must call destruct on the object.
   edm::ObjectWithDict setterObj = itFind->second.construct();

   //make it into the base class
   FWParameterSetterBase* p = static_cast<FWParameterSetterBase*>(setterObj.address());
   //Make a shared pointer to the base class that uses a destructor for the derived class, in order to match the above construct call.
   boost::shared_ptr<FWParameterSetterBase> ptr(p, boost::bind(&edm::TypeWithDict::destruct,itFind->second,setterObj.address(),true));
   return ptr;
}

/* Virtual function which sets widgets enabled state.*/
void
FWParameterSetterBase::setEnabled(bool)
{
}
