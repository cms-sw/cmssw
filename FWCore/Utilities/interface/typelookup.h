#ifndef FWCore_Utilities_typelookup_h
#define FWCore_Utilities_typelookup_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     typelookup
// 
/**\class typelookup typelookup.h FWCore/Utilities/interface/typelookup.h

 Description: Allow looking up a c++ type_info via its name

 Usage:
    This group of functions allow one to associate a string to a std::type_info.  In particular one
 can lookup an edm::TypeIDBase given a string.
 
 The system works by using the macros TYPELOOKUP_METHODS and DEFINE_TYPELOOKUP_REGISTRATION.
 TYPELOOKUP_METHODS(Tname): explicitly instantiates the template functions className classTypeInfo using
   the Tname value as both the C++ class and transforming it into the string name.
 
 DEFINE_TYPELOOKUP_REGISTRATION: registers the string name with the C++ class type.
 
 TYPELOOKUP_DATA_REG : sets both TYPELOOKUP_METHODS and DEFINE_TYPELOOKUP_REGISTRATION

 
 Example: You have a new data class called 'DummyData'.  Then to register that class with the system you
 place the lines
 
 #include "<where ever my class decleration lives>/interface/DummyData.h"
 
 TYPELOOKUP_DATA_REG(DummyData);
 
 into the file <where ever my class decleration lives>/src/T_EventSetup_DummyData.cc
 
 The actual name of the file that uses the 'TYPELOOKUP_DATA_REG' macro is not important.  The only important point
 the file that uses the 'TYPELOOKUP_DATA_REG' macro must be in the same library as the data class it is registering.
 
*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 20 14:26:21 CST 2010
// $Id: typelookup.h,v 1.2 2010/01/25 23:17:59 chrjones Exp $
//

// system include files
#include <typeinfo>
#include <utility>

// user include files

namespace edm {
   class TypeIDBase;
   
   namespace typelookup
   {
      /**Returns a std::type_info and a long lived const char* containing the name 
       associated with the string iClassName. If the string is not associated with a known 
       type then returns two null pointers */
       std::pair<const char*, const std::type_info*> findType(const char* iClassName);
      
      /**Returns the registered string (usually the class name) for the type T
       */
      template <typename T>
      const char* className();

      /**Returns the std::type_info for the class T.  This is done just by calling
       typeid(T).  So why bother? The call to typeid(T) requires one to include the
       header file which defines class T while the call to classTypeInfo<T>() does not.
       */
      template <typename T>
      const std::type_info& classTypeInfo();

      /**Used to create file static variables which register the string iTypeName to the C++ class
       type associated to the std::type_info.
       */
      class NameRegistrar {
      public:
         NameRegistrar(const char* iTypeName,const std::type_info& iInfo);
      };
      
   }
}

#define TYPELOOKUP_METHODS(Tname) \
namespace edm { namespace typelookup { \
template<> const char* className< Tname >() \
{ return #Tname ; } \
template<> const std::type_info& classTypeInfo< Tname > () \
{ return typeid( Tname ); } } }


#define EDM_TYPELOOKUP_SYM(x,y) EDM_TYPELOOKUP_SYM2(x,y)
#define EDM_TYPELOOKUP_SYM2(x,y) x ## y

#define DEFINE_TYPELOOKUP_REGISTRATION(type) \
static const edm::typelookup::NameRegistrar EDM_TYPELOOKUP_SYM(s_register , __LINE__ ) (edm::typelookup::className<type>(),typeid(type))

#define TYPELOOKUP_DATA_REG(_dataclass_) TYPELOOKUP_METHODS(_dataclass_) \
DEFINE_TYPELOOKUP_REGISTRATION(_dataclass_)


#endif
