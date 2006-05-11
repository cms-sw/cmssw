#ifndef Common_traits_h
#define Common_traits_h

/*----------------------------------------------------------------------

Definition of traits templates used in the EDM.  

$Id: traits.h,v 1.1 2006/02/07 07:01:51 wmtan Exp $

----------------------------------------------------------------------*/

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
}

#endif
