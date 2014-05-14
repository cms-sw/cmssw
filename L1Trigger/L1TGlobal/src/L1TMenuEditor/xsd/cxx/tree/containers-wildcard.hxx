// file      : xsd/cxx/tree/containers-wildcard.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_CONTAINERS_WILDCARD_HXX
#define XSD_CXX_TREE_CONTAINERS_WILDCARD_HXX

#include <set>
#include <string>

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/util/XMLString.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/containers.hxx> // iterator_adapter

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // one (for internal use only)
      //
      class element_one
      {
      public:
        ~element_one ()
        {
          if (x_)
            x_->release ();
        }

        explicit
        element_one (xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
        }

        element_one (const xercesc::DOMElement& x, xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
          set (x);
        }

        element_one (const element_one& x, xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
          if (x.present ())
            set (x.get ());
        }

        element_one&
        operator= (const element_one& x)
        {
          if (this == &x)
            return *this;

          if (x.present ())
            set (x.get ());
          else if (x_)
          {
            x_->release ();
            x_ = 0;
          }

          return *this;
        }

      public:
        const xercesc::DOMElement&
        get () const
        {
          return *x_;
        }

        xercesc::DOMElement&
        get ()
        {
          return *x_;
        }

        void
        set (const xercesc::DOMElement& x)
        {
          using xercesc::DOMElement;

          DOMElement* r (
            static_cast<DOMElement*> (
              doc_.importNode (const_cast<DOMElement*> (&x), true)));

          if (x_)
            x_->release ();

          x_ = r;
        }

        void
        set (xercesc::DOMElement* x)
        {
          assert (x->getOwnerDocument () == &doc_);

          if (x_)
            x_->release ();

          x_ = x;
        }

        bool
        present () const
        {
          return x_ != 0;
        }

      protected:
        xercesc::DOMElement* x_;
        xercesc::DOMDocument& doc_;
      };


      //
      //
      class element_optional
      {
      public:
        ~element_optional ()
        {
          if (x_)
            x_->release ();
        }

        explicit
        element_optional (xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
        }

        element_optional (const xercesc::DOMElement& x,
                          xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
          set (x);
        }

        element_optional (xercesc::DOMElement* x, xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
          set (x);
        }

        element_optional (const element_optional& x,
                          xercesc::DOMDocument& doc)
            : x_ (0), doc_ (doc)
        {
          if (x)
            set (*x);
        }

        element_optional&
        operator= (const xercesc::DOMElement& x)
        {
          if (x_ == &x)
            return *this;

          set (x);

          return *this;
        }

        element_optional&
        operator= (const element_optional& x)
        {
          if (this == &x)
            return *this;

          if (x)
            set (*x);
          else
            reset ();

          return *this;
        }

        // Pointer-like interface.
        //
      public:
        const xercesc::DOMElement*
        operator-> () const
        {
          return x_;
        }

        xercesc::DOMElement*
        operator-> ()
        {
          return x_;
        }

        const xercesc::DOMElement&
        operator* () const
        {
          return *x_;
        }

        xercesc::DOMElement&
        operator* ()
        {
          return *x_;
        }

        typedef void (element_optional::*bool_convertible) ();

        operator bool_convertible () const
        {
          return x_ != 0 ? &element_optional::true_ : 0;
        }

        // Get/set interface.
        //
      public:
        bool
        present () const
        {
          return x_ != 0;
        }

        const xercesc::DOMElement&
        get () const
        {
          return *x_;
        }

        xercesc::DOMElement&
        get ()
        {
          return *x_;
        }

        void
        set (const xercesc::DOMElement& x)
        {
          using xercesc::DOMElement;

          DOMElement* r (
            static_cast<DOMElement*> (
              doc_.importNode (const_cast<DOMElement*> (&x), true)));

          if (x_)
            x_->release ();

          x_ = r;
        }

        void
        set (xercesc::DOMElement* x)
        {
          assert (x->getOwnerDocument () == &doc_);

          if (x_)
            x_->release ();

          x_ = x;
        }

        void
        reset ()
        {
          if (x_)
            x_->release ();

          x_ = 0;
        }

      private:
        void
        true_ ()
        {
        }

      private:
        xercesc::DOMElement* x_;
        xercesc::DOMDocument& doc_;
      };

      // Comparison operators.
      //

      inline bool
      operator== (const element_optional& a, const element_optional& b)
      {
        return !a || !b
          ? a.present () == b.present ()
          : a->isEqualNode (&b.get ());
      }

      inline bool
      operator!= (const element_optional& a, const element_optional& b)
      {
        return !(a == b);
      }


      //
      //
      class element_sequence
      {
      protected:
        // This is a dangerously destructive automatic pointer. We are going
        // to use it in a controlled environment to save us a lot of coding.
        //
        struct ptr
        {
          ~ptr ()
          {
            if (x_)
              x_->release ();
          }

          explicit
          ptr (xercesc::DOMElement* x = 0)
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
              if (x_)
                x_->release ();

              x_ = y.x_;
              y.x_ = 0;
            }

            return *this;
          }

        public:
          xercesc::DOMElement&
          operator* () const
          {
            return *x_;
          }

          xercesc::DOMElement*
          get () const
          {
            return x_;
          }

        private:
          mutable xercesc::DOMElement* x_;
        };

        typedef std::vector<ptr> base_sequence;
        typedef base_sequence::iterator base_iterator;
        typedef base_sequence::const_iterator base_const_iterator;

      public:
        typedef xercesc::DOMElement        value_type;
        typedef xercesc::DOMElement*       pointer;
        typedef const xercesc::DOMElement* const_pointer;
        typedef xercesc::DOMElement&       reference;
        typedef const xercesc::DOMElement& const_reference;

        typedef
        iterator_adapter<base_sequence::iterator, xercesc::DOMElement>
        iterator;

        typedef
        iterator_adapter<base_sequence::const_iterator,
                         const xercesc::DOMElement>
        const_iterator;

        typedef
        iterator_adapter<base_sequence::reverse_iterator, xercesc::DOMElement>
        reverse_iterator;

        typedef
        iterator_adapter<base_sequence::const_reverse_iterator,
                         const xercesc::DOMElement>
        const_reverse_iterator;

        typedef base_sequence::size_type       size_type;
        typedef base_sequence::difference_type difference_type;
        typedef base_sequence::allocator_type  allocator_type;

      public:
        explicit
        element_sequence (xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
        }

        // DOMElement cannot be default-constructed.
        //
        // explicit
        // element_sequence (size_type n);

        element_sequence (size_type n,
                          const xercesc::DOMElement& x,
                          xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
          assign (n, x);
        }

        template <typename I>
        element_sequence (const I& begin, const I& end,
                          xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
          assign (begin, end);
        }

        element_sequence (const element_sequence& v,
                          xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
          v_.reserve (v.v_.size ());

          for (base_const_iterator i (v.v_.begin ()), e (v.v_.end ());
               i != e; ++i)
          {
            ptr p (static_cast<xercesc::DOMElement*> (
                     doc_.importNode (i->get (), true)));

            v_.push_back (p);
          }
        }

        element_sequence&
        operator= (const element_sequence& v)
        {
          if (this == &v)
            return *this;

          v_.assign (v.v_.size (), ptr ());

          base_iterator di (v_.begin ()), de (v_.end ());
          base_const_iterator si (v.v_.begin ()), se (v.v_.end ());

          for (; si != se && di != de; ++si, ++di)
          {
            ptr p (static_cast<xercesc::DOMElement*> (
                     doc_.importNode (si->get (), true)));
            *di = p;
          }

          return *this;
        }

      public:
        void
        assign (size_type n, const xercesc::DOMElement& x)
        {
          v_.assign (n, ptr ());

          for (base_iterator i (v_.begin ()), e (v_.end ()); i != e; ++i)
          {
            ptr p (static_cast<xercesc::DOMElement*> (
                     doc_.importNode (
                       const_cast<xercesc::DOMElement*> (&x), true)));
            *i = p;
          }
        }

        template <typename I>
        void
        assign (const I& begin, const I& end)
        {
          // This is not the fastest way to do it.
          //
          v_.clear ();

          for (I i (begin); i != end; ++i)
          {
            ptr p (static_cast<xercesc::DOMElement*> (
                     doc_.importNode (
                       const_cast<xercesc::DOMElement*> (&(*i)), true)));
            v_.push_back (p);
          }
        }

      public:
        // This version of resize can only be used to shrink the
        // sequence because DOMElement cannot be default-constructed.
        //
        void
        resize (size_type n)
        {
          assert (n <= v_.size ());
          v_.resize (n, ptr ());
        }

        void
        resize (size_type n, const xercesc::DOMElement& x)
        {
          size_type old (v_.size ());
          v_.resize (n, ptr ());

          if (old < n)
          {
            for (base_iterator i (v_.begin () + old), e (v_.end ());
                 i != e; ++i)
            {
              ptr p (static_cast<xercesc::DOMElement*> (
                       doc_.importNode (
                         const_cast<xercesc::DOMElement*> (&x), true)));
              *i = p;
            }
          }
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
        xercesc::DOMElement&
        operator[] (size_type n)
        {
          return *(v_[n]);
        }

        const xercesc::DOMElement&
        operator[] (size_type n) const
        {
          return *(v_[n]);
        }

        xercesc::DOMElement&
        at (size_type n)
        {
          return *(v_.at (n));
        }

        const xercesc::DOMElement&
        at (size_type n) const
        {
          return *(v_.at (n));
        }

        xercesc::DOMElement&
        front ()
        {
          return *(v_.front ());
        }

        const xercesc::DOMElement&
        front () const
        {
          return *(v_.front ());
        }

        xercesc::DOMElement&
        back ()
        {
          return *(v_.back ());
        }

        const xercesc::DOMElement&
        back () const
        {
          return *(v_.back ());
        }

      public:
        // Makes a deep copy.
        //
        void
        push_back (const xercesc::DOMElement& x)
        {
          ptr p (static_cast<xercesc::DOMElement*> (
                   doc_.importNode (
                     const_cast<xercesc::DOMElement*> (&x), true)));

          v_.push_back (p);
        }

        // Assumes ownership.
        //
        void
        push_back (xercesc::DOMElement* x)
        {
          assert (x->getOwnerDocument () == &doc_);
          v_.push_back (ptr (x));
        }

        void
        pop_back ()
        {
          v_.pop_back ();
        }

        // Makes a deep copy.
        //
        iterator
        insert (iterator position, const xercesc::DOMElement& x)
        {
          ptr p (static_cast<xercesc::DOMElement*> (
                   doc_.importNode (
                     const_cast<xercesc::DOMElement*> (&x), true)));

          return iterator (v_.insert (position.base (), p));
        }

        // Assumes ownership.
        //
        iterator
        insert (iterator position, xercesc::DOMElement* x)
        {
          assert (x->getOwnerDocument () == &doc_);
          return iterator (v_.insert (position.base (), ptr (x)));
        }

        void
        insert (iterator position, size_type n, const xercesc::DOMElement& x)
        {
          difference_type d (v_.end () - position.base ());
          v_.insert (position.base (), n, ptr ());

          for (base_iterator i (v_.end () - d); n != 0; --n)
          {
            ptr r (static_cast<xercesc::DOMElement*> (
                     doc_.importNode (
                       const_cast<xercesc::DOMElement*> (&x), true)));
            *(--i) = r;
          }
        }

        template <typename I>
        void
        insert (iterator position, const I& begin, const I& end)
        {
          // This is not the fastest way to do it.
          //
          if (begin != end)
          {
            base_iterator p (position.base ());

            for (I i (end);;)
            {
              --i;
              ptr r (static_cast<xercesc::DOMElement*> (
                       doc_.importNode (i->get (), true)));

              p = v_.insert (p, r);

              if (i == begin)
                break;
            }
          }
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

      public:
        // Note that the DOMDocument object of the two sequences being
	// swapped should be the same.
        //
        void
        swap (element_sequence& x)
        {
          assert (&doc_ == &x.doc_);
          v_.swap (x.v_);
        }

      private:
        base_sequence v_;
        xercesc::DOMDocument& doc_;
      };

      // Comparison operators.
      //

      inline bool
      operator== (const element_sequence& a, const element_sequence& b)
      {
        if (a.size () != b.size ())
          return false;

        element_sequence::const_iterator
          ai (a.begin ()), ae (a.end ()), bi (b.begin ());

        for (; ai != ae; ++ai, ++bi)
          if (!ai->isEqualNode (&(*bi)))
            return false;

        return true;
      }

      inline bool
      operator!= (const element_sequence& a, const element_sequence& b)
      {
        return !(a == b);
      }


      // Attribute set.
      //

      class attribute_set_common
      {
      protected:
        // Set entry. It can either act as a dangerously destructive
        // automatic pointer for DOMAttr or as an entry containing the
        // name we are searching for.
        //
        struct entry
        {
          ~entry ()
          {
            if (a_)
              a_->release ();
          }

          explicit
          entry (xercesc::DOMAttr* a)
              : a_ (a), ns_ (0), name_ (0)
          {
            ns_ = a->getNamespaceURI ();
            name_ = ns_ == 0 ? a->getName () : a->getLocalName ();
          }

          // Note: uses shallow copy.
          //
          explicit
          entry (const XMLCh* ns, const XMLCh* name)
              : a_ (0), ns_ (ns), name_ (name)
          {
          }

          entry (const entry& y)
              : a_ (y.a_), ns_ (y.ns_), name_ (y.name_)
          {
            // Yes, hostile takeover.
            //
            y.a_ = 0;
            y.ns_ = 0;
            y.name_ = 0;
          }

          entry&
          operator= (const entry& y)
          {
            if (this != &y)
            {
              // Yes, hostile takeover.
              //
              if (a_)
                a_->release ();

              a_ = y.a_;
              ns_ = y.ns_;
              name_ = y.name_;

              y.a_ = 0;
              y.ns_ = 0;
              y.name_ = 0;
            }

            return *this;
          }

        public:
          xercesc::DOMAttr&
          operator* () const
          {
            return *a_;
          }

          xercesc::DOMAttr*
          get () const
          {
            return a_;
          }

          const XMLCh*
          ns () const
          {
            return ns_;
          }

          const XMLCh*
          name () const
          {
            return name_;
          }

          void
          release ()
          {
            a_ = 0;
          }

        private:
          mutable xercesc::DOMAttr* a_;
          mutable const XMLCh* ns_;
          mutable const XMLCh* name_;
        };

        struct entry_cmp
        {
          bool
          operator() (const entry& a, const entry& b) const
          {
            using xercesc::XMLString;

            const XMLCh* ans (a.ns ());
            const XMLCh* bns (b.ns ());

            const XMLCh* an (a.name ());
            const XMLCh* bn (b.name ());

            if (ans == 0)
              return bns != 0
                ? true
                : (XMLString::compareString (an, bn) < 0);

            if (ans != 0 && bns == 0)
              return false;

            int r (XMLString::compareString (ans, bns));

            return r < 0
              ? true
              : (r > 0 ? false : XMLString::compareString (an, bn));
          }
        };

        typedef std::set<entry, entry_cmp> base_set;
        typedef base_set::iterator base_iterator;
        typedef base_set::const_iterator base_const_iterator;
      };

      template <typename C>
      class attribute_set: public attribute_set_common
      {
      public:
        typedef xercesc::DOMAttr        key_type;
        typedef xercesc::DOMAttr        value_type;
        typedef xercesc::DOMAttr*       pointer;
        typedef const xercesc::DOMAttr* const_pointer;
        typedef xercesc::DOMAttr&       reference;
        typedef const xercesc::DOMAttr& const_reference;

        typedef
        iterator_adapter<base_set::iterator, xercesc::DOMAttr>
        iterator;

        typedef
        iterator_adapter<base_set::const_iterator, const xercesc::DOMAttr>
        const_iterator;

        typedef
        iterator_adapter<base_set::reverse_iterator, xercesc::DOMAttr>
        reverse_iterator;

        typedef
        iterator_adapter<base_set::const_reverse_iterator,
                         const xercesc::DOMAttr>
        const_reverse_iterator;

        typedef base_set::size_type       size_type;
        typedef base_set::difference_type difference_type;
        typedef base_set::allocator_type  allocator_type;

      public:
        attribute_set (xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
        }

        template <typename I>
        attribute_set (const I& begin,
                       const I& end,
                       xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
          insert (begin, end);
        }

        attribute_set (const attribute_set& s, xercesc::DOMDocument& doc)
            : doc_ (doc)
        {
          // Can be done faster with the "hinted" insert.
          //
          insert (s.begin (), s.end ());
        }

        attribute_set&
        operator= (const attribute_set& s)
        {
          if (this == &s)
            return *this;

          // Can be done faster with the "hinted" insert.
          //
          clear ();
          insert (s.begin (), s.end ());

          return *this;
        }

      public:
        const_iterator
        begin () const
        {
          return const_iterator (s_.begin ());
        }

        const_iterator
        end () const
        {
          return const_iterator (s_.end ());
        }

        iterator
        begin ()
        {
          return iterator (s_.begin ());
        }

        iterator
        end ()
        {
          return iterator (s_.end ());
        }

        // reverse
        //

        const_reverse_iterator
        rbegin () const
        {
          return const_reverse_iterator (s_.rbegin ());
        }

        const_reverse_iterator
        rend () const
        {
          return const_reverse_iterator (s_.rend ());
        }

        reverse_iterator
        rbegin ()
        {
          return reverse_iterator (s_.rbegin ());
        }

        reverse_iterator
        rend ()
        {
          return reverse_iterator (s_.rend ());
        }

      public:
        size_type
        size () const
        {
          return s_.size ();
        }

        size_type
        max_size () const
        {
          return s_.max_size ();
        }

        bool
        empty () const
        {
          return s_.empty ();
        }

        void
        clear ()
        {
          s_.clear ();
        }

      public:
        // Makes a deep copy.
        //
        std::pair<iterator, bool>
        insert (const xercesc::DOMAttr& a)
        {
          entry e (static_cast<xercesc::DOMAttr*> (
                     doc_.importNode (
                       const_cast<xercesc::DOMAttr*> (&a), true)));

          std::pair<base_iterator, bool> r (s_.insert (e));

          return std::pair<iterator, bool> (iterator (r.first), r.second);
        }

        // Assumes ownership.
        //
        std::pair<iterator, bool>
        insert (xercesc::DOMAttr* a)
        {
          assert (a->getOwnerDocument () == &doc_);
          entry e (a);
          std::pair<base_iterator, bool> r (s_.insert (e));

          if (!r.second)
            e.release (); // Detach the attribute of insert failed.

          return std::pair<iterator, bool> (iterator (r.first), r.second);
        }

        // Makes a deep copy.
        //
        iterator
        insert (iterator position, const xercesc::DOMAttr& a)
        {
          entry e (static_cast<xercesc::DOMAttr*> (
                     doc_.importNode (
                       const_cast<xercesc::DOMAttr*> (&a), true)));

          return iterator (s_.insert (position.base (), e));
        }

        // Assumes ownership.
        //
        iterator
        insert (iterator position, xercesc::DOMAttr* a)
        {
          assert (a->getOwnerDocument () == &doc_);
          entry e (a);
          base_iterator r (s_.insert (position.base (), e));

          if (r->get () != a)
            e.release (); // Detach the attribute of insert failed.

          return iterator (r);
        }

        template <typename I>
        void
        insert (const I& begin, const I& end)
        {
          for (I i (begin); i != end; ++i)
          {
            entry e (static_cast<xercesc::DOMAttr*> (
                       doc_.importNode (
                         const_cast<xercesc::DOMAttr*> (&(*i)), true)));

            s_.insert (e);
          }
        }

      public:
        void
        erase (iterator position)
        {
          s_.erase (position.base ());
        }

        size_type
        erase (const std::basic_string<C>& name)
        {
          return s_.erase (entry (0, xml::string (name).c_str ()));
        }

        size_type
        erase (const std::basic_string<C>& namespace_,
               const std::basic_string<C>& name)
        {
          return s_.erase (entry (xml::string (namespace_).c_str (),
                                  xml::string (name).c_str ()));
        }

        size_type
        erase (const XMLCh* name)
        {
          return s_.erase (entry (0, name));
        }

        size_type
        erase (const XMLCh* namespace_, const XMLCh* name)
        {
          return s_.erase (entry (namespace_, name));
        }

        void
        erase (iterator begin, iterator end)
        {
          s_.erase (begin.base (), end.base ());
        }

      public:
        size_type
        count (const std::basic_string<C>& name) const
        {
          return s_.count (entry (0, xml::string (name).c_str ()));
        }

        size_type
        count (const std::basic_string<C>& namespace_,
               const std::basic_string<C>& name) const
        {
          return s_.count (entry (xml::string (namespace_).c_str (),
                                  xml::string (name).c_str ()));
        }

        size_type
        count (const XMLCh* name) const
        {
          return s_.count (entry (0, name));
        }

        size_type
        count (const XMLCh* namespace_, const XMLCh* name) const
        {
          return s_.count (entry (namespace_, name));
        }

        // find
        //

        iterator
        find (const std::basic_string<C>& name)
        {
          return iterator (s_.find (entry (0, xml::string (name).c_str ())));
        }

        iterator
        find (const std::basic_string<C>& namespace_,
              const std::basic_string<C>& name)
        {
          return iterator (
            s_.find (entry (xml::string (namespace_).c_str (),
                            xml::string (name).c_str ())));
        }

        iterator
        find (const XMLCh* name)
        {
          return iterator (s_.find (entry (0, name)));
        }

        iterator
        find (const XMLCh* namespace_, const XMLCh* name)
        {
          return iterator (s_.find (entry (namespace_, name)));
        }

        const_iterator
        find (const std::basic_string<C>& name) const
        {
          return const_iterator (
            s_.find (entry (0, xml::string (name).c_str ())));
        }

        const_iterator
        find (const std::basic_string<C>& namespace_,
               const std::basic_string<C>& name) const
        {
          return const_iterator (
            s_.find (entry (xml::string (namespace_).c_str (),
                            xml::string (name).c_str ())));
        }

        const_iterator
        find (const XMLCh* name) const
        {
          return const_iterator (s_.find (entry (0, name)));
        }

        const_iterator
        find (const XMLCh* namespace_, const XMLCh* name) const
        {
          return const_iterator (s_.find (entry (namespace_, name)));
        }

      public:
        // Note that the DOMDocument object of the two sets being
	// swapped should be the same.
        //
        void
        swap (attribute_set& x)
        {
          assert (&doc_ == &x.doc_);
          s_.swap (x.s_);
        }

      private:
        base_set s_;
        xercesc::DOMDocument& doc_;
      };

      // Comparison operators.
      //

      template <typename C>
      inline bool
      operator== (const attribute_set<C>& a, const attribute_set<C>& b)
      {
        if (a.size () != b.size ())
          return false;

        typename attribute_set<C>::const_iterator
          ai (a.begin ()), ae (a.end ()), bi (b.begin ());

        for (; ai != ae; ++ai, ++bi)
          if (!ai->isEqualNode (&(*bi)))
            return false;

        return true;
      }

      template <typename C>
      inline bool
      operator!= (const attribute_set<C>& a, const attribute_set<C>& b)
      {
        return !(a == b);
      }
    }
  }
}

#endif  // XSD_CXX_TREE_CONTAINERS_WILDCARD_HXX
