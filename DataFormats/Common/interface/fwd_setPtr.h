#ifndef DataFormats_Common_fwd_setPtr_h
#define DataFormats_Common_fwd_setPtr_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     setPtr
//
/**
 Description: Forward declare the standard setPtr functions

 Usage:

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 20 11:45:38 CEST 2007
//

// user include files
// system include files
#include <typeinfo>
#include <vector>
#include <list>
#include <set>
#include <deque>

// forward declarations
namespace edm {
  template <typename T, typename A>
  void
  setPtr(std::vector<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr);

  template <typename T, typename A>
  void
  setPtr(std::list<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr);

  template <typename T, typename A>
  void
  setPtr(std::deque<T, A> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr);

  template <typename T, typename A, typename Comp>
  void
  setPtr(std::set<T, A, Comp> const& obj,
         std::type_info const& iToType,
         unsigned long iIndex,
         void const*& oPtr);

}

#endif
