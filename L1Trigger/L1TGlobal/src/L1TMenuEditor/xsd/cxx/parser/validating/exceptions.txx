// file      : xsd/cxx/parser/validating/exceptions.txx
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
        // expected_attribute
        //
        template <typename C>
        expected_attribute<C>::
        ~expected_attribute ()
        {
        }

        template <typename C>
        expected_attribute<C>::
        expected_attribute (const std::basic_string<C>& expected_namespace,
                            const std::basic_string<C>& expected_name)
            : expected_namespace_ (expected_namespace),
              expected_name_ (expected_name)
        {
        }

        // unexpected_attribute
        //
        template <typename C>
        unexpected_attribute<C>::
        ~unexpected_attribute ()
        {
        }

        template <typename C>
        unexpected_attribute<C>::
        unexpected_attribute (const std::basic_string<C>& encountered_namespace,
                              const std::basic_string<C>& encountered_name)
            : encountered_namespace_ (encountered_namespace),
              encountered_name_ (encountered_name)
        {
        }

        // unexpected_characters
        //
        template <typename C>
        unexpected_characters<C>::
        ~unexpected_characters ()
        {
        }

        template <typename C>
        unexpected_characters<C>::
        unexpected_characters (const std::basic_string<C>& s)
            : characters_ (s)
        {
        }

        // invalid_value
        //
        template <typename C>
        invalid_value<C>::
        ~invalid_value ()
        {
        }

        template <typename C>
        invalid_value<C>::
        invalid_value (const C* type,
                       const std::basic_string<C>& value)
            : type_ (type), value_ (value)
        {
        }

        template <typename C>
        invalid_value<C>::
        invalid_value (const C* type,
                       const ro_string<C>& value)
            : type_ (type), value_ (value)
        {
        }

        template <typename C>
        invalid_value<C>::
        invalid_value (const std::basic_string<C>& type,
                       const std::basic_string<C>& value)
            : type_ (type), value_ (value)
        {
        }
      }
    }
  }
}
