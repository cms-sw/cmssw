// file      : xsd/cxx/tree/containers.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <ostream>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // one
      //
      template<typename T>
      one<T, false>::
      ~one ()
      {
        delete x_;
      }

      template<typename T>
      one<T, false>::
      one (flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
      }

      template<typename T>
      one<T, false>::
      one (const T& x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        set (x);
      }

      template<typename T>
      one<T, false>::
      one (std::auto_ptr<T> x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        set (x);
      }

      template<typename T>
      one<T, false>::
      one (const one<T, false>& x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        if (x.present ())
          set (x.get ());
      }

      template<typename T>
      one<T, false>& one<T, false>::
      operator= (const one<T, false>& x)
      {
        if (this == &x)
          return *this;

        if (x.present ())
          set (x.get ());
        else
        {
          delete x_;
          x_ = 0;
        }

        return *this;
      }

      template<typename T>
      void one<T, false>::
      set (const T& x)
      {
        // We always do a fresh copy because T may not be x's
        // dynamic type.
        //
        T* r (x._clone (flags_, container_));

        delete x_;
        x_ = r;
      }

      template<typename T>
      void one<T, false>::
      set (std::auto_ptr<T> x)
      {
        T* r (0);

        if (x.get () != 0)
        {
          if (x->_container () != container_)
            x->_container (container_);

          r = x.release ();
        }

        delete x_;
        x_ = r;
      }

      // optional
      //
      template <typename T>
      optional<T, false>::
      ~optional ()
      {
        delete x_;
      }

      template <typename T>
      optional<T, false>::
      optional (flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
      }

      template <typename T>
      optional<T, false>::
      optional (const T& x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        set (x);
      }

      template <typename T>
      optional<T, false>::
      optional (std::auto_ptr<T> x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        set (x);
      }

      template <typename T>
      optional<T, false>::
      optional (const optional<T, false>& x, flags f, container* c)
          : x_ (0), flags_ (f), container_ (c)
      {
        if (x)
          set (*x);
      }

      template <typename T>
      optional<T, false>& optional<T, false>::
      operator= (const T& x)
      {
        if (x_ == &x)
          return *this;

        set (x);

        return *this;
      }

      template <typename T>
      optional<T, false>& optional<T, false>::
      operator= (const optional<T, false>& x)
      {
        if (this == &x)
          return *this;

        if (x)
          set (*x);
        else
          reset ();

        return *this;
      }

      template <typename T>
      void optional<T, false>::
      set (const T& x)
      {
        // We always do a fresh copy because T may not be x's
        // dynamic type.
        //
        T* r (x._clone (flags_, container_));

        delete x_;
        x_ = r;
      }

      template <typename T>
      void optional<T, false>::
      set (std::auto_ptr<T> x)
      {
        T* r (0);

        if (x.get () != 0)
        {
          if (x->_container () != container_)
            x->_container (container_);

          r = x.release ();
        }

        delete x_;
        x_ = r;
      }

      template <typename T>
      void optional<T, false>::
      reset ()
      {
        delete x_;
        x_ = 0;
      }

      template <typename T>
      void optional<T, false>::
      true_ ()
      {
      }


      // optional
      //
      template <typename T>
      optional<T, true>::
      optional (const T& y, flags, container*)
          : present_ (false)
      {
        set (y);
      }

      template <typename T>
      optional<T, true>::
      optional (const optional<T, true>& y, flags, container*)
          : present_ (false)
      {
        if (y)
          set (*y);
      }

      template <typename T>
      optional<T, true>& optional<T, true>::
      operator= (const T& y)
      {
        if (&x_ == &y)
          return *this;

        set (y);

        return *this;
      }

      template <typename T>
      optional<T, true>& optional<T, true>::
      operator= (const optional<T, true>& y)
      {
        if (this == &y)
          return *this;

        if (y)
          set (*y);
        else
          reset ();

        return *this;
      }

      template <typename T>
      void optional<T, true>::
      true_ ()
      {
      }

      template <typename C, typename T, bool fund>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const optional<T, fund>& x)
      {
        if (x)
          os << *x;
        else
          os << bits::not_present<C> ();

        return os;
      }
    }
  }
}
