// file      : xsd/cxx/auto-array.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_AUTO_ARRAY_HXX
#define XSD_CXX_AUTO_ARRAY_HXX

#include <cstddef> // std::size_t

namespace xsd
{
  namespace cxx
  {
    template <typename T>
    struct std_deallocator
    {
      void
      deallocate (T* p)
      {
        delete[] p;
      }
    };

    // Simple automatic array. The second template parameter is
    // an optional deallocator type. If not specified, delete[]
    // is used.
    //
    template <typename T, typename D = std_deallocator<T> >
    struct auto_array
    {
      auto_array (T a[])
          : a_ (a), d_ (0)
      {
      }

      auto_array (T a[], D& d)
          : a_ (a), d_ (&d)
      {
      }

      ~auto_array ()
      {
        if (d_ != 0)
          d_->deallocate (a_);
        else
          delete[] a_;
      }

      T&
      operator[] (std::size_t index) const
      {
        return a_[index];
      }

      T*
      get () const
      {
        return a_;
      }

      T*
      release ()
      {
        T* tmp (a_);
        a_ = 0;
        return tmp;
      }

      void
      reset (T a[] = 0)
      {
        if (a_ != a)
        {
          if (d_ != 0)
            d_->deallocate (a_);
          else
            delete[] a_;

          a_ = a;
        }
      }

      typedef void (auto_array::*bool_convertible)();

      operator bool_convertible () const
      {
        return a_ ? &auto_array<T, D>::true_ : 0;
      }

    private:
      auto_array (const auto_array&);

      auto_array&
      operator= (const auto_array&);

    private:
      void
      true_ ();

    private:
      T* a_;
      D* d_;
    };

    template <typename T, typename D>
    void auto_array<T, D>::
    true_ ()
    {
    }
  }
}

#endif  // XSD_CXX_AUTO_ARRAY_HXX
