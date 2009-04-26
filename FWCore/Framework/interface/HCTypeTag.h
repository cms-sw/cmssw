#ifndef Framework_HCTypeTag_h
#define Framework_HCTypeTag_h
// -*- C++ -*-
//
// Package:     HeteroContainer
// Module:      HCTypeTag
// 
// Description: Base class for Tags that can specify a class Type.
//
// Usage:
//    This class is used in conjunction with HCTypeTagTemplate to specify
//    the class Type of another class.
//    To build a TypeTag you must initialize the correct HCTypeTagTemplate
//
//    Example
//       //Make a TypeTag for class Foo
//       HCTypeTag< MyGroup > fooTag = HCTypeTagTemplate< Foo, MyGroup >(); 
//
// Author:      Chris D. Jones
// Created:     Sun Sep 20 15:05:10 EDT 1998
// $Id: HCTypeTag.h,v 1.6 2008/01/16 14:20:14 chrjones Exp $
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/TypeIDBase.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
template< class Group >
class HCTypeTagRegistrar {
public:
   HCTypeTagRegistrar(const char* iTypeName,const std::type_info& iInfo);
};
         
template< class Group >
class HCTypeTag : public TypeIDBase
{
      // ---------- friend classes and functions ---------------
      friend class HCTypeTagRegistrar<Group>;
   public:
      // ---------- constants, enums and typedefs --------------

      // ---------- Constructors and destructor ----------------
      HCTypeTag() : m_name("") {}
      //virtual ~HCTypeTag();  

      // ---------- member functions ---------------------------

      // ---------- const member functions ---------------------
      const std::type_info& value() const { return typeInfo(); }
      const char*  name() const { return m_name; }

   /*
      bool operator==(const HCTypeTag< Group >& iRHS) const {
	 return m_value == iRHS.m_value; }
      bool operator!=(const HCTypeTag< Group >& iRHS) const {
	 return m_value != iRHS.m_value; }
      bool operator<(const HCTypeTag< Group >& iRHS) const {
	 return m_value < iRHS.m_value; }
      bool operator<=(const HCTypeTag< Group >& iRHS) const {
	 return m_value <= iRHS.m_value; }
      bool operator>(const HCTypeTag< Group >& iRHS) const {
	 return m_value > iRHS.m_value; }
      bool operator>=(const HCTypeTag< Group >& iRHS) const {
	 return m_value >= iRHS.m_value; }
*/
      ///find a type based on the types name, if not found will return default HCTypeTag
      static HCTypeTag<Group> findType(const std::string& iTypeName);
   protected:
      // ---------- protected member functions -----------------
      HCTypeTag(const std::type_info& iValue, const char* iName) :
	 TypeIDBase(iValue), m_name(iName) {}

      HCTypeTag(const TypeIDBase& iValue, const char* iName) :
         TypeIDBase(iValue), m_name(iName) {}

      // ---------- protected const member functions -----------

      // ---------- protected static member functions ----------
      static void registerName(const char* iTypeName,const std::type_info& iInfo);

   private:
      // ---------- Constructors and destructor ----------------
      //HCTypeTag(const HCTypeTag&); // use default

      // ---------- assignment operator(s) ---------------------
      //const HCTypeTag& operator=(const HCTypeTag&); // use default


      // ---------- data members -------------------------------
      const char*  m_name;


};
      }
   }
}
#endif
