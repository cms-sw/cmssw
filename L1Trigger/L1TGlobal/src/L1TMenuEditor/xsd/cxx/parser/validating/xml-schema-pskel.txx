// file      : xsd/cxx/parser/validating/xml-schema-pskel.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace validating
      {
        // any_type
        //

        template <typename C>
        bool any_type_pskel<C>::
        _start_element_impl (const ro_string<C>& ns,
                             const ro_string<C>& name,
                             const ro_string<C>* type)
        {
          this->_start_any_element (ns, name, type);
          this->complex_content<C>::context_.top ().any_ = true;
          return true;
        }

        template <typename C>
        bool any_type_pskel<C>::
        _end_element_impl (const ro_string<C>& ns, const ro_string<C>& name)
        {
          this->complex_content<C>::context_.top ().any_ = false;
          this->_end_any_element (ns, name);
          return true;
        }


        template <typename C>
        bool any_type_pskel<C>::
        _attribute_impl_phase_two (const ro_string<C>& ns,
                                   const ro_string<C>& name,
                                   const ro_string<C>& value)
        {
          this->_any_attribute (ns, name, value);
          return true;
        }

        template <typename C>
        bool any_type_pskel<C>::
        _characters_impl (const ro_string<C>& s)
        {
          this->_any_characters (s);
          return true;
        }

        // any_simple_type
        //

        template <typename C>
        bool any_simple_type_pskel<C>::
        _characters_impl (const ro_string<C>& s)
        {
          this->_any_characters (s);
          return true;
        }
      }
    }
  }
}
