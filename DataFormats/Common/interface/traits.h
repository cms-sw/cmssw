#ifndef Common_traits_h
#define Common_traits_h

/*----------------------------------------------------------------------

Definition of traits templates used in the EDM.  

$Id: traits.h,v 1.4 2006/08/10 23:34:53 wmtan Exp $

----------------------------------------------------------------------*/

#include <deque>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <string>

namespace edm
{
  //------------------------------------------------------------
  //
  // DoNotSortUponInsertion is a base class. Derive your own class X
  // from DoNotSortUponInsertion when: 
  //
  // 1. You want to use DetSetVector<X> as an EDProduct, but
  //
  // 2. You do *not* want the Event::put member template to cause the
  // DetSet<X> instances within the DetSetVector<X> to be sorted.
  //
  // DoNotSortUponInsertion has no behavior; it is used at compile
  // time to influence the behavior of Event::put.
  //
  // Usage:
  //    MyClass : public edm::DoNotSortUponInsertion { ... }
  //
  struct DoNotSortUponInsertion { };

#if ! __GNUC_PREREQ (3,4)
  //------------------------------------------------------------
  //
  // The trait struct template has_postinsert_trait<T> is used to
  // indicate whether or not the type T has a member function
  //
  //      void T::post_insert()
  //
  // This is used by Event::put to determine (at compile time) whether
  // or not such a function must be called.
  //
  // We assume the 'general case' for T is to not support post_insert.
  // Classes which do support post_insert must specialize this trait.
  //
  //------------------------------------------------------------

  template <class T>
  struct has_postinsert_trait
  {
    static bool const value = false;
  };

  //------------------------------------------------------------
  //
  // The trait struct template has_swap<T> is used to indicate
  // whether or not the type T has a member function
  //
  //   void T::swap(T&)
  //
  // This is used by Wrapper<T>::Wrapper(std::auto_ptr<T> x) to
  // determine (at compile time) whether a swap or a copy should be
  // used to set the state of the constructed Wrapper<T>.
  //
  // We provide partial specializations for standard library
  // collections here.  EDM container emplates are specialized in
  // their own headers.
  //------------------------------------------------------------

  template <class T>
  struct has_swap
  {
    static bool const value = false;
  };  

  template <class T, class A>
  struct has_swap<std::deque<T,A> >
  {
    static bool const value = true;
  };

  template <class T, class A>
  struct has_swap<std::list<T,A> >
  {
    static bool const value = true;
  };

  
  template <class K, class V, class C, class A>
  struct has_swap<std::map<K,V,C,A> >
  {
    static bool const value = true;
  };

  template <class K, class V, class C, class A>
  struct has_swap<std::multimap<K,V,C,A> >
  {
    static bool const value = true;
  };


  template <class V, class C, class A>
  struct has_swap<std::set<V,C,A> >
  {
    static bool const value = true;
  };


  template <class V, class C, class A>
  struct has_swap<std::multiset<V,C,A> >
  {
    static bool const value = true;
  };


  template <class T, class A>
  struct has_swap<std::vector<T,A> >
  {
    static bool const value = true;
  };

  template <>
  struct has_swap<std::string>
  {
    static bool const value = true;
  };
#endif

}

#endif
