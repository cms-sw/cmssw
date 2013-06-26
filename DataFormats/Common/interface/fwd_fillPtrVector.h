#ifndef DataFormats_Common_fwd_fillPtrVector_h
#define DataFormats_Common_fwd_fillPtrVector_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     fillPtrVector
//
/**

 Description: Forward declare standard edm::fillPtrVector functions

 Usage:

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 20 11:45:38 CEST 2007
//

// system include files

// user include files
#include <typeinfo>
#include <vector>
#include <list>
#include <set>
#include <deque>

// forward declarations
namespace edm {
  template <typename T, typename A>
  void
  fillPtrVector(std::vector<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr);
  
  template <typename T, typename A>
  void
  fillPtrVector(std::list<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr);

  template <typename T, typename A>
  void
  fillPtrVector(std::deque<T, A> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr);

  template <typename T, typename A, typename Comp>
  void
  fillPtrVector(std::set<T, A, Comp> const& obj,
                std::type_info const& iToType,
                std::vector<unsigned long> const& iIndicies,
                std::vector<void const*>& oPtr);
}

#endif
