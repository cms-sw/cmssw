// file      : xsd/cxx/tree/containers.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_CONTAINERS_HXX
#define XSD_CXX_TREE_CONTAINERS_HXX

#include <cstddef>   // std::ptrdiff_t
#include <string>
#include <vector>
#include <memory>    // std::auto_ptr
#include <iterator>  // std::iterator_traits
#include <algorithm> // std::equal, std::lexicographical_compare
#include <iosfwd>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // Test whether T is a fundamental C++ type.
      //

      template <typename T>
      struct fundamental_p
      {
        static const bool r = false;
      };

      // byte
      //
      template <>
      struct fundamental_p<signed char>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<unsigned char>
      {
        static const bool r = true;
      };

      // short
      //
      template <>
      struct fundamental_p<short>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<unsigned short>
      {
        static const bool r = true;
      };

      // int
      //
      template <>
      struct fundamental_p<int>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<unsigned int>
      {
        static const bool r = true;
      };

      // long
      //
      template <>
      struct fundamental_p<long>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<unsigned long>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<long long>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<unsigned long long>
      {
        static const bool r = true;
      };

      // bool
      //
      template <>
      struct fundamental_p<bool>
      {
        static const bool r = true;
      };

      // float
      //
      template <>
      struct fundamental_p<float>
      {
        static const bool r = true;
      };

      template <>
      struct fundamental_p<double>
      {
        static const bool r = true;
      };

      // one (for internal use only)
      //
      template <typename T, bool fund = fundamental_p<T>::r>
      class one;

      template <typename T>
      class one<T, false>
      {
      public:
        ~one ();

        one (flags, container*);

        one (const T&, flags, container*);

        one (std::auto_ptr<T>, flags, container*);

        one (const one&, flags, container*);

        one&
        operator= (const one&);

      public:
        const T&
        get () const
        {
          return *x_;
        }

        T&
        get ()
        {
          return *x_;
        }

        void
        set (const T&);

        void
        set (std::auto_ptr<T>);

        bool
        present () const
        {
          return x_ != 0;
        }

        std::auto_ptr<T>
        detach ()
        {
          T* x (x_);
          x->_container (0);
          x_ = 0;
          return std::auto_ptr<T> (x);
        }

      protected:
        T* x_;
        flags flags_;
        container* container_;
      };


      template <typename T>
      class one<T, true>
      {
      public:
        one (flags, container*)
            : present_ (false)
        {
        }

        one (const T& x, flags, container*)
            : x_ (x), present_ (true)
        {
        }

        one (const one& x, flags, container*)
            : x_ (x.x_), present_ (x.present_)
        {
        }

        one&
        operator= (const one& x)
        {
          if (this == &x)
            return *this;

          x_ = x.x_;
          present_ = x.present_;

          return *this;
        }

      public:
        const T&
        get () const
        {
          return x_;
        }

        T&
        get ()
        {
          return x_;
        }

        void
        set (const T& x)
        {
          x_ = x;
          present_ = true;
        }

        bool
        present () const
        {
          return present_;
        }

      protected:
        T x_;
        bool present_;
      };


      // Note that I cannot get rid of fund because of HP aCC3.
      //
      template <typename T, bool fund = fundamental_p<T>::r>
      class optional;

      template <typename T>
      class optional<T, false>
      {
      public:
        ~optional ();

        explicit
        optional (flags = 0, container* = 0);

        explicit
        optional (const T&, flags = 0, container* = 0);

        explicit
        optional (std::auto_ptr<T>, flags = 0, container* = 0);

        optional (const optional&, flags = 0, container* = 0);

        optional&
        operator= (const T&);

        optional&
        operator= (const optional&);

        // Pointer-like interface.
        //
      public:
        const T*
        operator-> () const
        {
          return x_;
        }

        T*
        operator-> ()
        {
          return x_;
        }

        const T&
        operator* () const
        {
          return *x_;
        }

        T&
        operator* ()
        {
          return *x_;
        }

        typedef optional self_; // Simplifier for Sun C++ 5.7.
        typedef void (self_::*bool_convertible) ();

        operator bool_convertible () const
        {
          return x_ != 0 ? &self_::true_ : 0;
        }

        // Get/set interface.
        //
      public:
        bool
        present () const
        {
          return x_ != 0;
        }

        const T&
        get () const
        {
          return *x_;
        }

        T&
        get ()
        {
          return *x_;
        }

        void
        set (const T&);

        void
        set (std::auto_ptr<T>);

        void
        reset ();

        std::auto_ptr<T>
        detach ()
        {
          T* x (x_);
          x->_container (0);
          x_ = 0;
          return std::auto_ptr<T> (x);
        }

      private:
        void
        true_ ();

      private:
        T* x_;
        flags flags_;
        container* container_;
      };


      //
      //
      template <typename T>
      class optional<T, true>
      {
      public:
        explicit
        optional (flags  = 0, container* = 0)
            : present_ (false)
        {
        }

        explicit
        optional (const T&, flags = 0, container* = 0);

        optional (const optional&, flags = 0, container* = 0);

        optional&
        operator= (const T&);

        optional&
        operator= (const optional&);

        // Pointer-like interface.
        //
      public:
        const T*
        operator-> () const
        {
          return &x_;
        }

        T*
        operator-> ()
        {
          return &x_;
        }

        const T&
        operator* () const
        {
          return get ();
        }

        T&
        operator* ()
        {
          return get ();
        }

        typedef optional self_; // Simplifier for Sun C++ 5.7.
        typedef void (self_::*bool_convertible) ();

        operator bool_convertible () const
        {
          return present () ? &self_::true_ : 0;
        }

        // Get/set interface.
        //
      public:
        bool
        present () const
        {
          return present_;
        }

        const T&
        get () const
        {
          return x_;
        }

        T&
        get ()
        {
          return x_;
        }

        void
        set (const T& y)
        {
          x_ = y;
          present_ = true;
        }

        void
        reset ()
        {
          present_ = false;
        }

      private:
        void
        true_ ();

      private:
        bool present_;
        T x_;
      };

      // Comparison operators.
      //

      template <typename T, bool fund>
      inline bool
      operator== (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return !a || !b ? a.present () == b.present () : *a == *b;
      }

      template <typename T, bool fund>
      inline bool
      operator!= (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return !(a == b);
      }

      template <typename T, bool fund>
      inline bool
      operator< (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return a && (!b || *a < *b);
      }

      template <typename T, bool fund>
      inline bool
      operator> (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return b < a;
      }

      template <typename T, bool fund>
      inline bool
      operator<= (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return !(a > b);
      }

      template <typename T, bool fund>
      inline bool
      operator>= (const optional<T, fund>& a, const optional<T, fund>& b)
      {
        return !(a < b);
      }

      // Provide an ostream insertion opretaor to prevent confusion from
      // the implicit bool conversion.
      //
      template <typename C, typename T, bool fund>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>&, const optional<T, fund>&);


      // Sequence.
      //

      // Note that I cannot get rid of 'fund' because HP aCC3 likes it
      // this way.
      //
      template <typename T, bool fund = fundamental_p<T>::r>
      class sequence;


      // Sun CC's <iterator> does not have iterator_traits. To overcome
      // this, we will wrap std::iterator_traits into our own and also
      // specialize it for pointer types. Since Sun CC uses pointer
      // for vector::iterator, it will use the specialization and won't
      // notice the std::iterator_traits.
      //
#ifndef _RWSTD_NO_CLASS_PARTIAL_SPEC
      template <typename I>
      struct iterator_traits
      {
        typedef
        typename std::iterator_traits<I>::iterator_category
        iterator_category;

        typedef
        typename std::iterator_traits<I>::value_type
        value_type;

        typedef
        typename std::iterator_traits<I>::difference_type
        difference_type;
      };
#else
      // The Pointer specialization does not work for reverse and
      // set iterators. But these iterators are user-dfined types
      // and have suitable typedefs that we can use.
      //
      template <typename I>
      struct iterator_traits
      {
        typedef typename I::iterator_category iterator_category;
        typedef typename I::value_type value_type;
        typedef typename I::difference_type difference_type;
      };

      template <typename T>
      struct iterator_traits<T*>
      {
        typedef std::random_access_iterator_tag iterator_category;
        typedef T value_type;
        typedef std::ptrdiff_t difference_type;
      };
#endif

      // Iterator adapter for complex types. It expects I to point to
      // a smart pointer-like object that has operator*() that returns
      // a refernce to a type static_cast'able to T and get() that
      // returns a pointer to a type static_cast'able to T.
      //

      template <typename I, typename T>
      struct iterator_adapter
      {
        typedef T value_type;
        typedef value_type& reference;
        typedef value_type* pointer;

        typedef
        typename iterator_traits<I>::iterator_category
        iterator_category;

        typedef
        typename iterator_traits<I>::difference_type
        difference_type;


      public:
        iterator_adapter ()
            : i_ () // i_ can be of a pointer type.
        {
        }

        // Allow iterator to const_iterator conversion.
        //
        template <typename J, typename T2>
        iterator_adapter (const iterator_adapter<J, T2>& j)
            : i_ (j.base ())
        {
        }

        explicit
        iterator_adapter (const I& i)
            : i_ (i)
        {
        }

      public:
        // Forward iterator requirements.
        //
        reference
        operator* () const
        {
          return static_cast<reference> (**i_);
        }

        pointer
        operator-> () const
        {
          return static_cast<pointer> (i_->get ());
        }

        iterator_adapter&
        operator++ ()
        {
          ++i_;
          return *this;
        }

        iterator_adapter
        operator++ (int)
        {
          iterator_adapter r (*this);
          ++i_;
          return r;
        }

        // Bidirectional iterator requirements.
        //
        iterator_adapter&
        operator-- ()
        {
          --i_;
          return *this;
        }

        iterator_adapter
        operator-- (int)
        {
          iterator_adapter r (*this);
          --i_;
          return r;
        }

        // Random access iterator requirements.
        //
        reference
        operator[] (difference_type n) const
        {
          return static_cast<reference> (*(i_[n]));
        }

        iterator_adapter&
        operator+= (difference_type n)
        {
          i_ += n;
          return *this;
        }

        iterator_adapter
        operator+ (difference_type n) const
        {
          return iterator_adapter (i_ + n);
        }

        iterator_adapter&
        operator-= (difference_type n)
        {
          i_ -= n;
          return *this;
        }

        iterator_adapter
        operator- (difference_type n) const
        {
          return iterator_adapter (i_ - n);
        }

      public:
        const I&
        base () const
        {
          return i_;
        }

      private:
        I i_;
      };

      // Note: We use different types for left- and right-hand-side
      // arguments to allow comparison between iterator and const_iterator.
      //

      // Forward iterator requirements.
      //
      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator== (const iterator_adapter<I, T1>& i,
                  const iterator_adapter<J, T2>& j)
      {
        return i.base () == j.base ();
      }

      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator!= (const iterator_adapter<I, T1>& i,
                  const iterator_adapter<J, T2>& j)
      {
        return i.base () != j.base ();
      }

      // Random access iterator requirements
      //
      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator< (const iterator_adapter<I, T1>& i,
                 const iterator_adapter<J, T2>& j)
      {
        return i.base () < j.base ();
      }

      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator> (const iterator_adapter<I, T1>& i,
                 const iterator_adapter<J, T2>& j)
      {
        return i.base () > j.base ();
      }

      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator<= (const iterator_adapter<I, T1>& i,
                  const iterator_adapter<J, T2>& j)
      {
        return i.base () <= j.base ();
      }

      template <typename I, typename J, typename T1, typename T2>
      inline bool
      operator>= (const iterator_adapter<I, T1>& i,
                  const iterator_adapter<J, T2>& j)
      {
        return i.base () >= j.base ();
      }

      template <typename I, typename J, typename T1, typename T2>
      inline typename iterator_adapter<I, T1>::difference_type
      operator- (const iterator_adapter<I, T1>& i,
                 const iterator_adapter<J, T2>& j)
      {
        return i.base () - j.base ();
      }

      template <typename I, typename T>
      inline iterator_adapter<I, T>
      operator+ (typename iterator_adapter<I, T>::difference_type n,
                 const iterator_adapter<I, T>& i)
      {
        return iterator_adapter<I, T> (i.base () + n);
      }

      //
      //
      class sequence_common
      {
      protected:
        // This is a dangerously destructive automatic pointer. We are going
        // to use it in a controlled environment to save us a lot of coding.
        //
        struct ptr
        {
          ~ptr ()
          {
            delete x_;
          }

          explicit
          ptr (type* x = 0)
              : x_ (x)
          {
          }

          ptr (const ptr& y)
              : x_ (y.x_)
          {
            // Yes, hostile takeover.
            //
            y.x_ = 0;
          }

          ptr&
          operator= (const ptr& y)
          {
            if (this != &y)
            {
              // Yes, hostile takeover.
              //
              delete x_;
              x_ = y.x_;
              y.x_ = 0;
            }

            return *this;
          }

        public:
          type&
          operator* () const
          {
            return *x_;
          }

          type*
          operator-> () const
          {
            return x_;
          }

          type*
          get () const
          {
            return x_;
          }

          type*
          release ()
          {
            type* x (x_);
            x_ = 0;
            return x;
          }

        private:
          mutable type* x_;
        };

      protected:
        typedef std::vector<ptr> base_sequence;
        typedef base_sequence::iterator base_iterator;
        typedef base_sequence::const_iterator base_const_iterator;

        typedef base_sequence::size_type       size_type;
        typedef base_sequence::difference_type difference_type;
        typedef base_sequence::allocator_type  allocator_type;

      protected:
        sequence_common (flags f, container* c)
            : flags_ (f), container_ (c)
        {
        }

        sequence_common (size_type n, const type& x, container* c)
            : flags_ (0), container_ (c)
        {
          assign (n, x);
        }

        template <typename I>
        sequence_common (const I& begin, const I& end, container* c)
            : flags_ (0), container_ (c)
        {
          assign (begin, end);
        }

        sequence_common (const sequence_common& v, flags f, container* c)
            : flags_ (f), container_ (c)
        {
          v_.reserve (v.v_.size ());

          for (base_const_iterator i (v.v_.begin ()), e (v.v_.end ());
               i != e; ++i)
          {
            ptr p ((**i)._clone (flags_, container_));
            v_.push_back (p);
          }
        }

      public:
        sequence_common&
        operator= (const sequence_common& v)
        {
          if (this == &v)
            return *this;

          v_.assign (v.v_.size (), ptr ());

          base_iterator di (v_.begin ()), de (v_.end ());
          base_const_iterator si (v.v_.begin ()), se (v.v_.end ());

          for (; si != se && di != de; ++si, ++di)
          {
            // We have no ptr_ref.
            //
            ptr p ((**si)._clone (flags_, container_));
            *di = p;
          }

          return *this;
        }

      public:
        size_type
        size () const
        {
          return v_.size ();
        }

        size_type
        max_size () const
        {
          return v_.max_size ();
        }

        size_type
        capacity () const
        {
          return v_.capacity ();
        }

        bool
        empty () const
        {
          return v_.empty ();
        }

        void
        reserve (size_type n)
        {
          v_.reserve (n);
        }

        void
        clear ()
        {
          v_.clear ();
        }

      protected:
        void
        assign (size_type n, const type& x)
        {
          v_.assign (n, ptr ());

          for (base_iterator i (v_.begin ()), e (v_.end ()); i != e; ++i)
          {
            ptr p (x._clone (flags_, container_));
            *i = p;
          }
        }

        template <typename I>
        void
        assign (const I& begin, const I& end)
        {
          // This is not the fastest way to do it. Also I's type may not
          // have _clone.
          //
          v_.clear ();

          for (I i (begin); i != end; ++i)
          {
            ptr p (i->_clone (flags_, container_));
            v_.push_back (p);
          }
        }

        void
        resize (size_type n, const type& x)
        {
          size_type old (v_.size ());
          v_.resize (n, ptr ());

          if (old < n)
          {
            for (base_iterator i (v_.begin () + old), e (v_.end ());
                 i != e; ++i)
            {
              ptr p (x._clone (flags_, container_));
              *i = p;
            }
          }
        }

        void
        insert (base_iterator p, size_type n, const type& x)
        {
          difference_type d (v_.end () - p);
          v_.insert (p, n, ptr ());

          for (base_iterator i (v_.end () - d); n != 0; --n)
          {
            ptr r (x._clone (flags_, container_));
            *(--i) = r;
          }
        }

        template <typename I>
        void
        insert (base_iterator p, const I& begin, const I& end)
        {
          // This is not the fastest way to do it. Also I's type may not
          // have _clone.
          //
          if (begin != end)
          {
            for (I i (end);;)
            {
              --i;
              ptr r (i->_clone (flags_, container_));
              p = v_.insert (p, r);

              if (i == begin)
                break;
            }
          }
        }

      protected:
        flags flags_;
        container* container_;
        base_sequence v_;
      };

      //
      //
      template <typename T>
      class sequence<T, false>: public sequence_common
      {
      protected:
        // For IBM XL C++ 8.0.
        //
        typedef sequence_common::ptr ptr;

      public:
        typedef T        value_type;
        typedef T*       pointer;
        typedef const T* const_pointer;
        typedef T&       reference;
        typedef const T& const_reference;

        typedef
        iterator_adapter<base_sequence::iterator, T>
        iterator;

        typedef
        iterator_adapter<base_sequence::const_iterator, const T>
        const_iterator;

        typedef
        iterator_adapter<base_sequence::reverse_iterator, T>
        reverse_iterator;

        typedef
        iterator_adapter<base_sequence::const_reverse_iterator, const T>
        const_reverse_iterator;

        typedef sequence_common::size_type       size_type;
        typedef sequence_common::difference_type difference_type;
        typedef sequence_common::allocator_type  allocator_type;

      public:
        explicit
        sequence (flags f = 0, container* c = 0)
            : sequence_common (f, c)
        {
        }

        // The first version causes trouble on IBM XL C++ 7.0 when
        // a type does not have the default c-tor. While the second
        // breaks VC++ 8.0 when using dllexport (it appears to
        // instantiate everything instead of only what's used).
        //
#ifdef _MSC_VER
        explicit
        sequence (size_type n, const T& x = T (), container* c = 0)
            : sequence_common (n, x, c)
        {
        }
#else
        explicit
        sequence (size_type n, container* c = 0)
            : sequence_common (n, T (), c)
        {
        }

        sequence (size_type n, const T& x, container* c = 0)
            : sequence_common (n, x, c)
        {
        }
#endif

        template <typename I>
        sequence (const I& begin, const I& end, container* c = 0)
            : sequence_common (begin, end, c)
        {
        }

        sequence (const sequence& v, flags f = 0, container* c = 0)
            : sequence_common (v, f, c)
        {
        }

      public:
        void
        assign (size_type n, const T& x)
        {
          sequence_common::assign (n, x);
        }

        template <typename I>
        void
        assign (const I& begin, const I& end)
        {
          sequence_common::assign (begin, end);
        }

      public:
        // The first version causes trouble on IBM XL C++ 7.0 when
        // a type does not have the default c-tor. While the second
        // breaks VC++ 8.0 when using dllexport (it appears to
        // instantiate everything instead of only what's used).
        //
#ifdef _MSC_VER
        void
        resize (size_type n, const T& x = T ())
        {
          sequence_common::resize (n, x);
        }
#else
        void
        resize (size_type n)
        {
          sequence_common::resize (n, T ());
        }

        void
        resize (size_type n, const T& x)
        {
          sequence_common::resize (n, x);
        }
#endif

      public:
        const_iterator
        begin () const
        {
          return const_iterator (v_.begin ());
        }

        const_iterator
        end () const
        {
          return const_iterator (v_.end ());
        }

        iterator
        begin ()
        {
          return iterator (v_.begin ());
        }

        iterator
        end ()
        {
          return iterator (v_.end ());
        }

        // reverse
        //

        const_reverse_iterator
        rbegin () const
        {
          return const_reverse_iterator (v_.rbegin ());
        }

        const_reverse_iterator
        rend () const
        {
          return const_reverse_iterator (v_.rend ());
        }

        reverse_iterator
        rbegin ()
        {
          return reverse_iterator (v_.rbegin ());
        }

        reverse_iterator
        rend ()
        {
          return reverse_iterator (v_.rend ());
        }

      public:
        T&
        operator[] (size_type n)
        {
          return static_cast<T&> (*(v_[n]));
        }

        const T&
        operator[] (size_type n) const
        {
          return static_cast<const T&> (*(v_[n]));
        }

        T&
        at (size_type n)
        {
          return static_cast<T&> (*(v_.at (n)));
        }

        const T&
        at (size_type n) const
        {
          return static_cast<const T&> (*(v_.at (n)));
        }

        T&
        front ()
        {
          return static_cast<T&> (*(v_.front ()));
        }

        const T&
        front () const
        {
          return static_cast<const T&> (*(v_.front ()));
        }

        T&
        back ()
        {
          return static_cast<T&> (*(v_.back ()));
        }

        const T&
        back () const
        {
          return static_cast<const T&> (*(v_.back ()));
        }

      public:
        void
        push_back (const T& x)
        {
          v_.push_back (ptr (x._clone (flags_, container_)));
        }

        void
        push_back (std::auto_ptr<T> x)
        {
          if (x->_container () != container_)
            x->_container (container_);

          v_.push_back (ptr (x.release ()));
        }

        void
        pop_back ()
        {
          v_.pop_back ();
        }

        std::auto_ptr<T>
        detach_back (bool pop = true)
        {
          ptr& p (v_.back ());
          p->_container (0);
          T* x (static_cast<T*> (p.release ()));

          if (pop)
            v_.pop_back ();

          return std::auto_ptr<T> (x);
        }

        iterator
        insert (iterator position, const T& x)
        {
          return iterator (
            v_.insert (
              position.base (), ptr (x._clone (flags_, container_))));
        }

        iterator
        insert (iterator position, std::auto_ptr<T> x)
        {
          if (x->_container () != container_)
            x->_container (container_);

          return iterator (v_.insert (position.base (), ptr (x.release ())));
        }

        void
        insert (iterator position, size_type n, const T& x)
        {
          sequence_common::insert (position.base (), n, x);
        }

        template <typename I>
        void
        insert (iterator position, const I& begin, const I& end)
        {
          sequence_common::insert (position.base (), begin, end);
        }

        iterator
        erase (iterator position)
        {
          return iterator (v_.erase (position.base ()));
        }

        iterator
        erase (iterator begin, iterator end)
        {
          return iterator (v_.erase (begin.base (), end.base ()));
        }

        iterator
        detach (iterator position, std::auto_ptr<T>& r, bool erase = true)
        {
          ptr& p (*position.base ());
          p->_container (0);
          std::auto_ptr<T> tmp (static_cast<T*> (p.release ()));
          r = tmp;

          if (erase)
            return iterator (v_.erase (position.base ()));
          else
            return ++position;
        }

        // Note that the container object of the two sequences being
	// swapped should be the same.
        //
        void
        swap (sequence& x)
        {
          assert (container_ == x.container_);
          v_.swap (x.v_);
        }
      };


      // Specialization for fundamental types.
      //
      template <typename T>
      class sequence<T, true>: public std::vector<T>
      {
        typedef std::vector<T> base_sequence;

      public:
        explicit
        sequence (flags = 0, container* = 0)
        {
        }

        explicit
        sequence (typename base_sequence::size_type n,
                  const T& x = T (),
                  container* = 0)
            : base_sequence (n, x)
        {
        }

        template <typename I>
        sequence (const I& begin, const I& end, container* = 0)
            : base_sequence (begin, end)
        {
        }

        sequence (const sequence& s, flags = 0, container* = 0)
            : base_sequence (s)
        {
        }
      };


      // Comparison operators.
      //

      template <typename T, bool fund>
      inline bool
      operator== (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return (a.size () == b.size ()
                && std::equal (a.begin (), a.end (), b.begin ()));
      }

      template <typename T, bool fund>
      inline bool
      operator!= (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return !(a == b);
      }

      template <typename T, bool fund>
      inline bool
      operator< (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return std::lexicographical_compare (a.begin (), a.end (),
                                             b.begin (), b.end ());
      }

      template <typename T, bool fund>
      inline bool
      operator> (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return b < a;
      }

      template <typename T, bool fund>
      inline bool
      operator<= (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return !(a > b);
      }

      template <typename T, bool fund>
      inline bool
      operator>= (const sequence<T, fund>& a, const sequence<T, fund>& b)
      {
        return !(a < b);
      }

      // Note that the container object of the two sequences being
      // swapped should be the same.
      //
      template <typename T, bool fund>
      inline void
      swap (sequence<T, fund>& x, sequence<T, fund>& y)
      {
        x.swap (y);
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/containers.txx>

#endif  // XSD_CXX_TREE_CONTAINERS_HXX
