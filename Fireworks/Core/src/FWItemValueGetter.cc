// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemValueGetter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 30 16:15:43 EST 2008
// $Id: FWItemValueGetter.cc,v 1.3 2009/01/23 21:35:43 amraktad Exp $
//

// system include files
#include <sstream>
#include "Reflex/Object.h"
#include "Reflex/Base.h"

// user include files
#include "Fireworks/Core/interface/FWItemValueGetter.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
ROOT::Reflex::Member
recursiveFindMember(const std::string& iName,
                    const ROOT::Reflex::Type& iType)
{
   using namespace ROOT::Reflex;

   Member temp = iType.MemberByName(iName);
   if(temp) {return temp;}

   //try all base classes
   for(Base_Iterator it = iType.Base_Begin(), itEnd = iType.Base_End();
       it != itEnd;
       ++it) {
      temp = recursiveFindMember(iName,it->ToType());
      if(temp) {break;}
   }
   return temp;
}

namespace {
   template <class T>
   std::string valueToString(const std::string& iName,
                             const std::string& iUnit,
                             const Reflex::Object& iObject,
                             const Reflex::Member& iMember) {
      std::stringstream s;
      s.setf(std::ios_base::fixed,std::ios_base::floatfield);
      s.precision(1);
      T temp;
      iMember.Invoke(iObject,temp);
      s<<iName <<": "<<temp<<" "<<iUnit;
      return s.str();
   }

   typedef std::string (*FunctionType)(const std::string&, const std::string&,const Reflex::Object&, const Reflex::Member&);
   typedef std::map<std::string, FunctionType> TypeToStringMap;

   template<typename T>
   static void addToStringMap(TypeToStringMap& iMap) {
      iMap[typeid(T).name()]=valueToString<T>;
   }

   template <class T>
   double valueToDouble(const Reflex::Object& iObj, const Reflex::Member& iMember) {
      T temp;
      iMember.Invoke(iObj,temp);
      return temp;
   }

   typedef double (*DoubleFunctionType)(const Reflex::Object&, const Reflex::Member&);
   typedef std::map<std::string, DoubleFunctionType> TypeToDoubleMap;

   template<typename T>
   static void addToDoubleMap(TypeToDoubleMap& iMap) {
      iMap[typeid(T).name()]=valueToDouble<T>;
   }

}

static
std::string
stringValueFor(const ROOT::Reflex::Object& iObj, 
               const ROOT::Reflex::Member& iMember,
               const std::string& iUnit) {
   static TypeToStringMap s_map;
   if(s_map.empty() ) {
      addToStringMap<float>(s_map);
      addToStringMap<double>(s_map);
   }
   Reflex::Type returnType = iMember.TypeOf().ReturnType().FinalType();
   

   TypeToStringMap::iterator itFound =s_map.find(returnType.TypeInfo().name());
   if(itFound == s_map.end()) {
      //std::cout <<" could not print because type is "<<iObj.TypeOf().TypeInfo().name()<<std::endl;
      return std::string();
   }

   return itFound->second(iMember.Name(),iUnit,iObj,iMember);
}

static
double
doubleValueFor(const ROOT::Reflex::Object& iObj, const ROOT::Reflex::Member& iMember) {
   static TypeToDoubleMap s_map;
   if(s_map.empty() ) {
      addToDoubleMap<float>(s_map);
      addToDoubleMap<double>(s_map);
   }

   const Reflex::Type returnType = iMember.TypeOf().ReturnType().FinalType();

   //std::cout << val.TypeOf().TypeInfo().name()<<std::endl;
   TypeToDoubleMap::iterator itFound =s_map.find(returnType.TypeInfo().name());
   if(itFound == s_map.end()) {
      //std::cout <<" could not print because type is "<<iObj.TypeOf().TypeInfo().name()<<std::endl;
      return -999.0;
   }

   return itFound->second(iObj,iMember);
}


//
// constructors and destructor
//
FWItemValueGetter::FWItemValueGetter(const ROOT::Reflex::Type& iType,
                                     const std::vector<std::pair<std::string, std::string> >& iFindValueFrom) :
   m_type(iType)
{
   using namespace ROOT::Reflex;
   for(std::vector<std::pair<std::string,std::string> >::const_iterator it = iFindValueFrom.begin(), itEnd=iFindValueFrom.end();
       it != itEnd;
       ++it) {
      //std::cout <<" trying function "<<*it<<std::endl;
      Member temp = recursiveFindMember(it->first,iType);
      if(temp) {
         if(0==temp.FunctionParameterSize(true)) {
            //std::cout <<"    FOUND "<<temp.Name()<<std::endl;
            //std::cout <<"     in type "<<temp.DeclaringType().Name(SCOPED)<<std::endl;
            m_memberFunction = temp;
            m_unit = it->second;
            break;
         }
      }
   }
}

// FWItemValueGetter::FWItemValueGetter(const FWItemValueGetter& rhs)
// {
//    // do actual copying here;
// }

//FWItemValueGetter::~FWItemValueGetter()
//{
//}

//
// assignment operators
//
// const FWItemValueGetter& FWItemValueGetter::operator=(const FWItemValueGetter& rhs)
// {
//   //An exception safe implementation is
//   FWItemValueGetter temp(rhs);
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
double
FWItemValueGetter::valueFor(const void* iObject) const
{
   ROOT::Reflex::Object temp(m_type,
                             const_cast<void*>(iObject));
   ROOT::Reflex::Object obj= temp.CastObject(m_memberFunction.DeclaringType());
   return ::doubleValueFor(obj,m_memberFunction);
}

std::string
FWItemValueGetter::stringValueFor(const void* iObject) const
{
   ROOT::Reflex::Object temp(m_type,
                             const_cast<void*>(iObject));
   ROOT::Reflex::Object obj= temp.CastObject(m_memberFunction.DeclaringType());

   return ::stringValueFor(obj,m_memberFunction,m_unit);
}

bool
FWItemValueGetter::isValid() const
{
   return bool(m_memberFunction);
}

std::string
FWItemValueGetter::valueName() const
{
   return m_memberFunction.Name();
}

const std::string&
FWItemValueGetter::unit() const
{
   return m_unit;
}

//
// static member functions
//
