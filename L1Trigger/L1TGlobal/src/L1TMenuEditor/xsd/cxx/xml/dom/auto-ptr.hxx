// file      : xsd/cxx/xml/dom/auto-ptr.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_AUTO_PTR_HXX
#define XSD_CXX_XML_DOM_AUTO_PTR_HXX

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        // Simple auto_ptr version that calls release() instead of delete.
        //

        template <typename T>
        struct remove_c
        {
          typedef T r;
        };

        template <typename T>
        struct remove_c<const T>
        {
          typedef T r;
        };

        template <typename T>
        struct auto_ptr_ref
        {
          T* x_;

          explicit
          auto_ptr_ref (T* x)
              : x_ (x)
          {
          }
        };

        template <typename T>
        struct auto_ptr
        {
          ~auto_ptr ()
          {
            reset ();
          }

          explicit
          auto_ptr (T* x = 0)
              : x_ (x)
          {
          }

          auto_ptr (auto_ptr& y)
              : x_ (y.release ())
          {
          }

          template <typename T2>
          auto_ptr (auto_ptr<T2>& y)
              : x_ (y.release ())
          {
          }

          auto_ptr (auto_ptr_ref<T> r)
              : x_ (r.x_)
          {
          }

          auto_ptr&
          operator= (auto_ptr& y)
          {
            if (x_ != y.x_)
              reset (y.release ());

            return *this;
          }

          template <typename T2>
          auto_ptr&
          operator= (auto_ptr<T2>& y)
          {
            if (x_ != y.x_)
              reset (y.release ());

            return *this;
          }

          auto_ptr&
          operator= (auto_ptr_ref<T> r)
          {
            if (r.x_ != x_)
              reset (r.x_);

            return *this;
          }

          template <typename T2>
          operator auto_ptr_ref<T2> ()
          {
            return auto_ptr_ref<T2> (release ());
          }

          template <typename T2>
          operator auto_ptr<T2> ()
          {
            return auto_ptr<T2> (release ());
          }

        public:
          T&
          operator* () const
          {
            return *x_;
          }

          T*
          operator-> () const
          {
            return x_;
          }

          T*
          get () const
          {
            return x_;
          }

          T*
          release ()
          {
            T* x (x_);
            x_ = 0;
            return x;
          }

          void
          reset (T* x = 0)
          {
            if (x_)
              const_cast<typename remove_c<T>::r*> (x_)->release ();

            x_ = x;
          }

        private:
          T* x_;
        };
      }
    }
  }
}

#endif // XSD_CXX_XML_DOM_AUTO_PTR_HXX
