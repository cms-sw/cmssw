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
// $Id: HCTypeTag.h,v 1.2 2005/06/23 19:59:30 wmtan Exp $
//
//

// system include files
#include <string>

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace heterocontainer {
template< class Group >
class HCTypeTag
{
      // ---------- friend classes and functions ---------------

   public:
      // ---------- constants, enums and typedefs --------------
      enum { kDefaultValue = 0 };

      // ---------- Constructors and destructor ----------------
      HCTypeTag() : m_value(kDefaultValue), m_name(0) {}
      //virtual ~HCTypeTag();  

      // ---------- member functions ---------------------------

      // ---------- const member functions ---------------------
      unsigned int value() const { return m_value; }
      const char*  name() const { return m_name; }

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

      ///find a type based on the types name, if not found will return default HCTypeTag
      static HCTypeTag<Group> findType(const std::string& iTypeName);
   protected:
      // ---------- protected member functions -----------------
      HCTypeTag(unsigned int iValue, const char* iName) :
	 m_value(iValue), m_name(iName) {}

      // ---------- protected const member functions -----------

      // ---------- protected static member functions ----------
      static unsigned int nextValue(const char* iTypeName);

   private:
      // ---------- Constructors and destructor ----------------
      //HCTypeTag(const HCTypeTag&); // use default

      // ---------- assignment operator(s) ---------------------
      //const HCTypeTag& operator=(const HCTypeTag&); // use default


      // ---------- data members -------------------------------
      unsigned int m_value;
      const char*  m_name;


};

      }
   }
}
#endif /* Framework_HCTypeTag_h */
