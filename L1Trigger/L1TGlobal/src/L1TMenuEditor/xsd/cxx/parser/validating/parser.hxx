// file      : xsd/cxx/parser/validating/parser.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_VALIDATING_PARSER_HXX
#define XSD_CXX_PARSER_VALIDATING_PARSER_HXX

#include <stack>
#include <cstddef> // std::size_t
#include <cstring> // std::memcpy

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        //
        //
        template <typename C>
        struct empty_content: parser_base<C>
        {
          // These functions are called when wildcard content
          // is encountered. Use them to handle mixed content
          // models, any/anyAttribute, and anyType/anySimpleType.
          // By default these functions do nothing.
          //

          // The type argument is a type name and namespace from the
          // xsi:type attribute in the form "<name> <namespace>" with
          // the space and namespace part absent if the type does not
          // have a namespace or 0 if xsi:type is not present.
          //
          virtual void
          _start_any_element (const ro_string<C>& ns,
                              const ro_string<C>& name,
                              const ro_string<C>* type);

          virtual void
          _end_any_element (const ro_string<C>& ns,
                            const ro_string<C>& name);

          virtual void
          _any_attribute (const ro_string<C>& ns,
                          const ro_string<C>& name,
                          const ro_string<C>& value);

          virtual void
          _any_characters (const ro_string<C>&);


          //
          //
          virtual bool
          _start_element_impl (const ro_string<C>&,
                               const ro_string<C>&,
                               const ro_string<C>*);

          virtual bool
          _end_element_impl (const ro_string<C>&,
                             const ro_string<C>&);

          virtual bool
          _attribute_impl (const ro_string<C>&,
                           const ro_string<C>&,
                           const ro_string<C>&);

          virtual bool
          _characters_impl (const ro_string<C>&);


          //
          //
          virtual void
          _start_element (const ro_string<C>&,
                          const ro_string<C>&,
                          const ro_string<C>*);

          virtual void
          _end_element (const ro_string<C>&,
                        const ro_string<C>&);

          virtual void
          _attribute (const ro_string<C>&,
                      const ro_string<C>&,
                      const ro_string<C>&);

          virtual void
          _characters (const ro_string<C>&);


          //
          //
          virtual void
          _expected_element (const C* expected_ns,
                             const C* expected_name);

          virtual void
          _expected_element (const C* expected_ns,
                             const C* expected_name,
                             const ro_string<C>& encountered_ns,
                             const ro_string<C>& encountered_name);

          virtual void
          _unexpected_element (const ro_string<C>& ns,
                               const ro_string<C>& name);

          virtual void
          _expected_attribute (const C* expected_ns,
                               const C* expected_name);

          virtual void
          _unexpected_attribute (const ro_string<C>& ns,
                                 const ro_string<C>& name,
                                 const ro_string<C>& value);

          virtual void
          _unexpected_characters (const ro_string<C>&);
        };


        //
        //
        template <typename C>
        struct simple_content: empty_content<C>
        {
          //
          //
          virtual void
          _attribute (const ro_string<C>& ns,
                      const ro_string<C>& name,
                      const ro_string<C>& value);

          virtual void
          _characters (const ro_string<C>&);

          //
          //
          virtual bool
          _attribute_impl (const ro_string<C>&,
                           const ro_string<C>&,
                           const ro_string<C>&);

          //
          //
          virtual void
          _pre_impl ();

          virtual void
          _post_impl ();


          // Implementation callbacks.
          //
          virtual void
          _pre_a_validate ();

          virtual void
          _post_a_validate ();


          // Attribute validation: during phase one we are searching for
          // matching attributes (Structures, section 3.4.4, clause 2.1).
          // During phase two we are searching for attribute wildcards
          // (section 3.4.4, clause 2.2). Both phases run across
          // inheritance hierarchy from derived to base for extension
          // only. Both functions return true if the match was found and
          // validation has been performed.
          //
          virtual bool
          _attribute_impl_phase_one (const ro_string<C>& ns,
                                     const ro_string<C>& name,
                                     const ro_string<C>& value);

          virtual bool
          _attribute_impl_phase_two (const ro_string<C>& ns,
                                     const ro_string<C>& name,
                                     const ro_string<C>& value);
        };


        //
        //
        template <typename C>
        struct complex_content: empty_content<C>
        {
          //
          //
          virtual void
          _start_element (const ro_string<C>& ns,
                          const ro_string<C>& name,
                          const ro_string<C>* type);

          virtual void
          _end_element (const ro_string<C>& ns,
                        const ro_string<C>& name);

          virtual void
          _attribute (const ro_string<C>& ns,
                      const ro_string<C>& name,
                      const ro_string<C>& value);

          virtual void
          _characters (const ro_string<C>&);

          //
          //
          virtual bool
          _attribute_impl (const ro_string<C>&,
                           const ro_string<C>&,
                           const ro_string<C>&);

          //
          //
          virtual void
          _pre_impl ();

          virtual void
          _post_impl ();


          // Implementation callbacks.
          //
          virtual void
          _pre_e_validate ();

          virtual void
          _post_e_validate ();

          virtual void
          _pre_a_validate ();

          virtual void
          _post_a_validate ();


          // Attribute validation: during phase one we are searching for
          // matching attributes (Structures, section 3.4.4, clause 2.1).
          // During phase two we are searching for attribute wildcards
          // (section 3.4.4, clause 2.2). Both phases run across
          // inheritance hierarchy from derived to base for extension
          // only. Both functions return true if the match was found and
          // validation has been performed.
          //
          virtual bool
          _attribute_impl_phase_one (const ro_string<C>& ns,
                                     const ro_string<C>& name,
                                     const ro_string<C>& value);

          virtual bool
          _attribute_impl_phase_two (const ro_string<C>& ns,
                                     const ro_string<C>& name,
                                     const ro_string<C>& value);
        protected:
          struct state
          {
            state ()
                : any_ (false), depth_ (0), parser_ (0)
            {
            }

            bool any_;
            std::size_t depth_;
            parser_base<C>* parser_;
          };

          // Optimized state stack for non-recursive case (one element).
          //
          struct state_stack
          {
            state_stack ()
                : size_ (0)
            {
            }

            void
            push (const state& s)
            {
              if (size_ > 0)
                rest_.push (top_);

              top_ = s;
              ++size_;
            }

            void
            pop ()
            {
              if (size_ > 1)
              {
                top_ = rest_.top ();
                rest_.pop ();
              }

              --size_;
            }

            const state&
            top () const
            {
              return top_;
            }

            state&
            top ()
            {
              return top_;
            }

            state&
            under_top ()
            {
              return rest_.top ();
            }

          private:
            state top_;
            std::stack<state> rest_;
            std::size_t size_;
          };

          state_stack context_;
        };

        // Base for xsd:list.
        //
        template <typename C>
        struct list_base: simple_content<C>
        {
          virtual void
          _xsd_parse_item (const ro_string<C>&) = 0;

          virtual void
          _pre_impl ();

          virtual void
          _characters (const ro_string<C>&);

          virtual void
          _post_impl ();

        protected:
          std::basic_string<C> buf_;
        };
      }

      // POD stack with pre-allocated first element. You may
      // need to pad your elements to get the proper alignment.
      //
      struct pod_stack
      {
        ~pod_stack ()
        {
          delete[] data_;
        }

        pod_stack (std::size_t element_size, void* first_element)
            : el_size_ (element_size), first_ (first_element),
              data_ (0), size_ (0), capacity_ (0)
        {
        }

      public:
        void
        pop ()
        {
          --size_;
        }

        void
        push ()
        {
          if (size_ > capacity_)
            grow ();

          ++size_;
        }

        void*
        top ()
        {
          return size_ == 1 ? first_ : data_ + (size_ - 1) * el_size_;
        }

        void*
        under_top ()
        {
          return size_ == 2 ? first_ : data_ + (size_ - 2) * el_size_;
        }

        std::size_t
        element_size () const
        {
          return el_size_;
        }

      private:
        void
        grow ()
        {
          std::size_t c (capacity_ ? capacity_ * 2 : 8);
          char* d (new char[c * el_size_]);

          if (size_ > 1)
            std::memcpy (d, data_, (size_ - 1) * el_size_);

          delete[] data_;

          data_ = d;
          capacity_ = c;
        }

      private:
        std::size_t el_size_;
        void* first_;
        char* data_;
        std::size_t size_;
        std::size_t capacity_;
      };

      namespace validating
      {
        // Validation state stack for the 'all' particle.
        //
        struct all_stack
        {
          all_stack (std::size_t n, unsigned char* first)
              : stack_ (n, first)
          {
          }

          void
          push ()
          {
            stack_.push ();

            unsigned char* p (static_cast<unsigned char*> (stack_.top ()));

            for (std::size_t i (0); i < stack_.element_size (); ++i)
              p[i] = 0;
          }

          void
          pop ()
          {
            stack_.pop ();
          }

          unsigned char*
          top ()
          {
            return static_cast<unsigned char*> (stack_.top ());
          }

        private:
          pod_stack stack_;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/parser.txx>

#endif  // XSD_CXX_PARSER_VALIDATING_PARSER_HXX
