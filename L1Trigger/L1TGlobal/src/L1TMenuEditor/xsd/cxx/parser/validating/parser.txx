// file      : xsd/cxx/parser/validating/parser.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <cassert>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/validating/exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {

        // empty_content
        //


        template <typename C>
        void empty_content<C>::
        _start_any_element (const ro_string<C>&,
                            const ro_string<C>&,
                            const ro_string<C>*)
        {
        }

        template <typename C>
        void empty_content<C>::
        _end_any_element (const ro_string<C>&,
                          const ro_string<C>&)
        {
        }

        template <typename C>
        void empty_content<C>::
        _any_attribute (const ro_string<C>&,
                        const ro_string<C>&,
                        const ro_string<C>&)
        {
        }

        template <typename C>
        void empty_content<C>::
        _any_characters (const ro_string<C>&)
        {
        }

        //
        //
        template <typename C>
        bool empty_content<C>::
        _start_element_impl (const ro_string<C>&,
                             const ro_string<C>&,
                             const ro_string<C>*)
        {
          return false;
        }

        template <typename C>
        bool empty_content<C>::
        _end_element_impl (const ro_string<C>&,
                           const ro_string<C>&)
        {
          return false;
        }

        template <typename C>
        bool empty_content<C>::
        _attribute_impl (const ro_string<C>&,
                         const ro_string<C>&,
                         const ro_string<C>&)
        {
          return false;
        }

        template <typename C>
        bool empty_content<C>::
        _characters_impl (const ro_string<C>&)
        {
          return false;
        }

        //
        //
        template <typename C>
        void empty_content<C>::
        _start_element (const ro_string<C>& ns,
                        const ro_string<C>& name,
                        const ro_string<C>* type)
        {
          if (!this->_start_element_impl (ns, name, type))
            this->_unexpected_element (ns, name);
        }

        template <typename C>
        void empty_content<C>::
        _end_element (const ro_string<C>& ns,
                      const ro_string<C>& name)
        {
          if (!this->_end_element_impl (ns, name))
            this->_unexpected_element (ns, name);
        }

        template <typename C>
        void empty_content<C>::
        _attribute (const ro_string<C>& ns,
                    const ro_string<C>& name,
                    const ro_string<C>& value)
        {
          // Weed out special attributes: xsi:type, xsi:nil,
          // xsi:schemaLocation and noNamespaceSchemaLocation.
          // See section 3.2.7 in Structures for details.
          //
          if (ns == xml::bits::xsi_namespace<C> () &&
              (name == xml::bits::type<C> () ||
               name == xml::bits::nil_lit<C> () ||
               name == xml::bits::schema_location<C> () ||
               name == xml::bits::no_namespace_schema_location<C> ()))
            return;

          // Also some parsers (notably Xerces-C++) supplies us with
          // namespace-prefix mapping attributes.
          //
          if (ns == xml::bits::xmlns_namespace<C> ())
            return;

          if (!this->_attribute_impl (ns, name, value))
            this->_unexpected_attribute (ns, name, value);
        }

        template <typename C>
        void empty_content<C>::
        _characters (const ro_string<C>& s)
        {
          if (!this->_characters_impl (s))
            this->_unexpected_characters (s);
        }

        //
        //
        template <typename C>
        void empty_content<C>::
        _expected_element (const C* ex_ns, const C* ex_name)
        {
          throw expected_element<C> (ex_ns, ex_name);
        }

        template <typename C>
        void empty_content<C>::
        _expected_element (const C* ex_ns,
                           const C* ex_name,
                           const ro_string<C>& en_ns,
                           const ro_string<C>& en_name)
        {
          throw expected_element<C> (ex_ns, ex_name, en_ns, en_name);
        }

        template <typename C>
        void empty_content<C>::
        _unexpected_element (const ro_string<C>& ns,
                             const ro_string<C>& name)
        {
          throw unexpected_element<C> (ns, name);
        }

        template <typename C>
        void empty_content<C>::
        _expected_attribute (const C* ex_ns, const C* ex_name)
        {
          throw expected_attribute<C> (ex_ns, ex_name);
        }

        template <typename C>
        void empty_content<C>::
        _unexpected_attribute (const ro_string<C>& ns,
                               const ro_string<C>& name,
                               const ro_string<C>&)
        {
          throw unexpected_attribute<C> (ns, name);
        }

        template <typename C>
        void empty_content<C>::
        _unexpected_characters (const ro_string<C>& s)
        {
          throw unexpected_characters<C> (s);
        }


        // simple_content
        //

        template <typename C>
        void simple_content<C>::
        _attribute (const ro_string<C>& ns,
                    const ro_string<C>& name,
                    const ro_string<C>& value)
        {
          // Weed out special attributes: xsi:type, xsi:nil,
          // xsi:schemaLocation and xsi:noNamespaceSchemaLocation.
          // See section 3.2.7 in Structures for details.
          //
          if (ns == xml::bits::xsi_namespace<C> () &&
              (name == xml::bits::type<C> () ||
               name == xml::bits::nil_lit<C> () ||
               name == xml::bits::schema_location<C> () ||
               name == xml::bits::no_namespace_schema_location<C> ()))
            return;

          // Also some parsers (notably Xerces-C++) supplies us with
          // namespace-prefix mapping attributes.
          //
          if (ns == xml::bits::xmlns_namespace<C> ())
            return;

          if (!this->_attribute_impl (ns, name, value))
            this->_unexpected_attribute (ns, name, value);
        }

        template <typename C>
        void simple_content<C>::
        _characters (const ro_string<C>& str)
        {
          if (!this->_characters_impl (str))
          {
            // Mixed content is implemented in the generated code
            // by overriding _characters_impl and forwarding to
            // _any_characters.
            //

            // Scan the string for any non-whitespace characters
            // (Structures, section 3.4.4, clause 1.3).
            //
            for (typename ro_string<C>::size_type i (0), e (str.size ());
                 i < e; ++i)
            {
              C c (str[i]);

              if (c != C (0x20) && // space
                  c != C (0x0D) && // carriage return
                  c != C (0x09) && // tab
                  c != C (0x0A))
                this->_unexpected_characters (str);
            }
          }
        }

        template <typename C>
        void simple_content<C>::
        _pre_impl ()
        {
          this->_pre ();
          _pre_a_validate ();
        }

        template <typename C>
        void simple_content<C>::
        _post_impl ()
        {
          _post_a_validate ();
          this->_post ();
        }

        template <typename C>
        void simple_content<C>::
        _pre_a_validate ()
        {
        }

        template <typename C>
        void simple_content<C>::
        _post_a_validate ()
        {
        }

        template <typename C>
        bool simple_content<C>::
        _attribute_impl (const ro_string<C>& ns,
                         const ro_string<C>& name,
                         const ro_string<C>& value)
        {
          return _attribute_impl_phase_one (ns, name, value) ||
            _attribute_impl_phase_two (ns, name, value);
        }

        template <typename C>
        bool simple_content<C>::
        _attribute_impl_phase_one (const ro_string<C>&,
                                   const ro_string<C>&,
                                   const ro_string<C>&)
        {
          return false;
        }

        template <typename C>
        bool simple_content<C>::
        _attribute_impl_phase_two (const ro_string<C>&,
                                   const ro_string<C>&,
                                   const ro_string<C>&)
        {
          return false;
        }


        // complex_content
        //


        template <typename C>
        void complex_content<C>::
        _start_element (const ro_string<C>& ns,
                        const ro_string<C>& name,
                        const ro_string<C>* type)
        {
          state& s (context_.top ());

          if (s.depth_++ > 0)
          {
            if (s.any_)
              this->_start_any_element (ns, name, type);
            else if (s.parser_)
              s.parser_->_start_element (ns, name, type);
          }
          else
          {
            if (!this->_start_element_impl (ns, name, type))
              this->_unexpected_element (ns, name);
            else if (s.parser_ != 0)
              s.parser_->_pre_impl ();
          }
        }

        template <typename C>
        void complex_content<C>::
        _end_element (const ro_string<C>& ns,
                      const ro_string<C>& name)
        {
          // To understand what's going on here it is helpful to think of
          // a "total depth" as being the sum of individual depths over
          // all elements.
          //

          if (context_.top ().depth_ == 0)
          {
            state& s (context_.under_top ()); // One before last.

            if (--s.depth_ > 0)
            {
              // Indirect recursion.
              //
              if (s.parser_)
                s.parser_->_end_element (ns, name);
            }
            else
            {
              // Direct recursion.
              //
              assert (this == s.parser_);

              this->_post_impl ();

              if (!this->_end_element_impl (ns, name))
                assert (false);
            }
          }
          else
          {
            state& s (context_.top ());

            if (--s.depth_ > 0)
            {
              if (s.any_)
                this->_end_any_element (ns, name);
              else if (s.parser_)
                s.parser_->_end_element (ns, name);
            }
            else
            {
              if (s.parser_ != 0 && !s.any_)
                s.parser_->_post_impl ();

              if (!this->_end_element_impl (ns, name))
                this->_unexpected_element (ns, name);
            }
          }
        }

        template <typename C>
        void complex_content<C>::
        _attribute (const ro_string<C>& ns,
                    const ro_string<C>& name,
                    const ro_string<C>& value)
        {
          // Weed out special attributes: xsi:type, xsi:nil,
          // xsi:schemaLocation and xsi:noNamespaceSchemaLocation.
          // See section 3.2.7 in Structures for details.
          //
          if (ns == xml::bits::xsi_namespace<C> () &&
              (name == xml::bits::type<C> () ||
               name == xml::bits::nil_lit<C> () ||
               name == xml::bits::schema_location<C> () ||
               name == xml::bits::no_namespace_schema_location<C> ()))
            return;

          // Also some parsers (notably Xerces-C++) supplies us with
          // namespace-prefix mapping attributes.
          //
          if (ns == xml::bits::xmlns_namespace<C> ())
            return;

          const state& s (context_.top ());

          if (s.depth_ > 0)
          {
            if (s.any_)
              this->_any_attribute (ns, name, value);
            else if (s.parser_)
              s.parser_->_attribute (ns, name, value);
          }
          else
          {
            if (!this->_attribute_impl (ns, name, value))
              this->_unexpected_attribute (ns, name, value);
          }
        }

        template <typename C>
        void complex_content<C>::
        _characters (const ro_string<C>& str)
        {
          const state& s (context_.top ());

          if (s.depth_ > 0)
          {
            if (s.any_)
              this->_any_characters (str);
            else if (s.parser_)
              s.parser_->_characters (str);
          }
          else
          {
            if (!this->_characters_impl (str))
            {
              // Mixed content is implemented in the generated code
              // by overriding _characters_impl and forwarding to
              // _any_characters.
              //

              // Scan the string for any non-whitespace characters
              // (Structures, section 3.4.4, clause 1.3).
              //
              for (typename ro_string<C>::size_type i (0), e (str.size ());
                   i < e; ++i)
              {
                C c (str[i]);

                if (c != C (0x20) && // space
                    c != C (0x0D) && // carriage return
                    c != C (0x09) && // tab
                    c != C (0x0A))
                  this->_unexpected_characters (str);
              }
            }
          }
        }

        template <typename C>
        void complex_content<C>::
        _pre_impl ()
        {
          context_.push (state ());
          this->_pre ();
          _pre_a_validate ();
          _pre_e_validate ();
        }

        template <typename C>
        void complex_content<C>::
        _post_impl ()
        {
          _post_e_validate ();
          _post_a_validate ();
          this->_post ();
          context_.pop ();
        }

        template <typename C>
        void complex_content<C>::
        _pre_e_validate ()
        {
        }

        template <typename C>
        void complex_content<C>::
        _post_e_validate ()
        {
        }

        template <typename C>
        void complex_content<C>::
        _pre_a_validate ()
        {
        }

        template <typename C>
        void complex_content<C>::
        _post_a_validate ()
        {
        }

        template <typename C>
        bool complex_content<C>::
        _attribute_impl (const ro_string<C>& ns,
                         const ro_string<C>& name,
                         const ro_string<C>& value)
        {
          return _attribute_impl_phase_one (ns, name, value) ||
            _attribute_impl_phase_two (ns, name, value);
        }

        template <typename C>
        bool complex_content<C>::
        _attribute_impl_phase_one (const ro_string<C>&,
                                   const ro_string<C>&,
                                   const ro_string<C>&)
        {
          return false;
        }

        template <typename C>
        bool complex_content<C>::
        _attribute_impl_phase_two (const ro_string<C>&,
                                   const ro_string<C>&,
                                   const ro_string<C>&)
        {
          return false;
        }


        // list_base
        //
        namespace bits
        {
          // Find first non-space character.
          //
          template <typename C>
          typename ro_string<C>::size_type
          find_ns (const C* s,
                   typename ro_string<C>::size_type size,
                   typename ro_string<C>::size_type pos)
          {
            while (pos < size &&
                   (s[pos] == C (0x20) || s[pos] == C (0x0A) ||
                    s[pos] == C (0x0D) || s[pos] == C (0x09)))
              ++pos;

            return pos < size ? pos : ro_string<C>::npos;
          }

          // Find first space character.
          //
          template <typename C>
          typename ro_string<C>::size_type
          find_s (const C* s,
                  typename ro_string<C>::size_type size,
                  typename ro_string<C>::size_type pos)
          {
            while (pos < size &&
                   s[pos] != C (0x20) && s[pos] != C (0x0A) &&
                   s[pos] != C (0x0D) && s[pos] != C (0x09))
              ++pos;

            return pos < size ? pos : ro_string<C>::npos;
          }
        }

        // Relevant XML Schema Part 2: Datatypes sections: 4.2.1.2, 4.3.6.
        //

        template <typename C>
        void list_base<C>::
        _pre_impl ()
        {
          simple_content<C>::_pre_impl ();
          buf_.clear ();
        }

        template <typename C>
        void list_base<C>::
        _characters (const ro_string<C>& s)
        {
          typedef typename ro_string<C>::size_type size_type;

          const C* data (s.data ());
          size_type size (s.size ());

          // Handle the previous chunk if we start with a ws.
          //
          if (!buf_.empty () &&
              (data[0] == C (0x20) || data[0] == C (0x0A) ||
               data[0] == C (0x0D) || data[0] == C (0x09)))
          {
            ro_string<C> tmp (buf_); // Private copy ctor.
            _xsd_parse_item (tmp);
            buf_.clear ();
          }

          // Traverse the data while logically collapsing spaces.
          //
          for (size_type i (bits::find_ns (data, size, 0));
               i != ro_string<C>::npos;)
          {
            size_type j (bits::find_s (data, size, i));

            if (j != ro_string<C>::npos)
            {
              if (buf_.empty ())
              {
                ro_string<C> tmp (data + i, j - i); // Private copy ctor.
                _xsd_parse_item (tmp);
              }
              else
              {
                // Assemble the first item in str from buf_ and s.
                //
                std::basic_string<C> str;
                str.swap (buf_);
                str.append (data + i, j - i);
                ro_string<C> tmp (str); // Private copy ctor.
                _xsd_parse_item (tmp);
              }

              i = bits::find_ns (data, size, j);
            }
            else
            {
              // Last fragment, append it to the buf_.
              //
              buf_.append (data + i, size - i);
              break;
            }
          }
        }

        template <typename C>
        void list_base<C>::
        _post_impl ()
        {
          // Handle the last item.
          //
          if (!buf_.empty ())
          {
            ro_string<C> tmp (buf_); // Private copy ctor.
            _xsd_parse_item (tmp);
          }

          simple_content<C>::_post_impl ();
        }
      }
    }
  }
}
