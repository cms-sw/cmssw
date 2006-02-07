#ifndef Common_traits_h
#define Common_traits_h

/*----------------------------------------------------------------------

Definition of traits templates used in the EDM.  

$Id: traits.h,v 1.1 2005/09/30 21:11:18 paterno Exp $

----------------------------------------------------------------------*/

namespace edm
{
  //------------------------------------------------------------

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
