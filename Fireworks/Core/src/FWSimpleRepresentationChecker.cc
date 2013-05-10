// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSimpleRepresentationChecker
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 10:54:28 EST 2008
// $Id: FWSimpleRepresentationChecker.cc,v 1.6 2010/06/02 22:55:42 chrjones Exp $
//

// system include files
#include <iostream>
#include "TClass.h"
#include "Reflex/Base.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"

#include "Fireworks/Core/interface/FWRepresentationInfo.h"

#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/Core/interface/FWItemAccessorBase.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSimpleRepresentationChecker::FWSimpleRepresentationChecker(const std::string& iTypeName,
                                                             const std::string& iPurpose,
                                                             unsigned int iBitPackedViews,
                                                             bool iRepresentsSubPart) :
   FWRepresentationCheckerBase(iPurpose,iBitPackedViews,iRepresentsSubPart),
   m_typeidName(iTypeName)
{
}

// FWSimpleRepresentationChecker::FWSimpleRepresentationChecker(const FWSimpleRepresentationChecker& rhs)
// {
//    // do actual copying here;
// }

FWSimpleRepresentationChecker::~FWSimpleRepresentationChecker()
{
}

//
// assignment operators
//
// const FWSimpleRepresentationChecker& FWSimpleRepresentationChecker::operator=(const FWSimpleRepresentationChecker& rhs)
// {
//   //An exception safe implementation is
//   FWSimpleRepresentationChecker temp(rhs);
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
static bool inheritsFrom(const Reflex::Type& iChild,
                         const std::string& iParentTypeName,
                         unsigned int& distance) {
   if(iChild.TypeInfo().name() == iParentTypeName) {
      return true;
   }
   if(iChild.BaseSize() == 0) {
      return false;
   }
   ++distance;
   for(Reflex::Base_Iterator it = iChild.Base_Begin(),
                                   itEnd = iChild.Base_End();
       it != itEnd;
       ++it) {
      if(inheritsFrom(it->ToType(),iParentTypeName,distance)) {
         return true;
      }
   }
   --distance;
   return false;
}

FWRepresentationInfo
FWSimpleRepresentationChecker::infoFor(const std::string& iTypeName) const
{
   unsigned int distance=1;

   FWItemAccessorFactory factory;
   //std::cout<<"checker infoFor"<<iTypeName<<std::endl;
   TClass* clss = TClass::GetClass(iTypeName.c_str());
   //Class could be unknown if the dictionary for it has not been loaded
   if(0==clss || 0==clss->GetTypeInfo()) {
      return FWRepresentationInfo();
   }
   boost::shared_ptr<FWItemAccessorBase> accessor = factory.accessorFor(clss);

   const TClass* modelClass = accessor->modelType();
   //std::cout <<"   "<<modelClass->GetName()<<" "<< bool(modelClass == clss)<< std::endl;

   if(0==modelClass || 0 == modelClass->GetTypeInfo()) {
      //some containers e.g. vector<int> do not have known TClasses for their elements
      // or the contained type may be unknown to ROOT
      return FWRepresentationInfo();
   }
   Reflex::Type modelType =
      Reflex::Type::ByTypeInfo( *(modelClass->GetTypeInfo()));
   //see if the modelType inherits from our type

   if(inheritsFrom(modelType,m_typeidName,distance) ) {
      return FWRepresentationInfo(purpose(),distance,bitPackedViews(), representsSubPart());
   }
   return FWRepresentationInfo();
}

//
// static member functions
//
