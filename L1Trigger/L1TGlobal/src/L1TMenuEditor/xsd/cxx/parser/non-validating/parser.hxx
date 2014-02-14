// file      : xsd/cxx/parser/non-validating/parser.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_NON_VALIDATING_PARSER_HXX
#define XSD_CXX_PARSER_NON_VALIDATING_PARSER_HXX

#include <stack>
#include <string>
#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace non_validating
      {
        //
        //
        template <typename C>
        struct empty_content: parser_base<C>
        {
          // The _*_any_* functions are called when wildcard content
          // is encountered. Use them to handle mixed content models,
          // any/anyAttribute, and anyType/anySimpleType. By default
          // these functions do nothing.
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
          _characters (const ro_string<C>& s);
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
          virtual void
          _pre_impl ();

          virtual void
          _post_impl ();

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
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/non-validating/parser.txx>

#endif  // XSD_CXX_PARSER_NON_VALIDATING_PARSER_HXX
