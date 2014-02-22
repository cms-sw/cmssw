// file      : xsd/cxx/parser/schema-exceptions.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // expected_element
      //
      template <typename C>
      expected_element<C>::
      ~expected_element ()
      {
      }

      template <typename C>
      expected_element<C>::
      expected_element (const std::basic_string<C>& expected_namespace,
                        const std::basic_string<C>& expected_name)
          : expected_namespace_ (expected_namespace),
            expected_name_ (expected_name)
      {
      }

      template <typename C>
      expected_element<C>::
      expected_element (const std::basic_string<C>& expected_namespace,
                        const std::basic_string<C>& expected_name,
                        const std::basic_string<C>& encountered_namespace,
                        const std::basic_string<C>& encountered_name)
          : expected_namespace_ (expected_namespace),
            expected_name_ (expected_name),
            encountered_namespace_ (encountered_namespace),
            encountered_name_ (encountered_name)
      {
      }

      // unexpected_element
      //
      template <typename C>
      unexpected_element<C>::
      ~unexpected_element ()
      {
      }

      template <typename C>
      unexpected_element<C>::
      unexpected_element (const std::basic_string<C>& encountered_namespace,
                          const std::basic_string<C>& encountered_name)
          : encountered_namespace_ (encountered_namespace),
            encountered_name_ (encountered_name)
      {
      }

      // dynamic_type
      //
      template <typename C>
      dynamic_type<C>::
      ~dynamic_type () throw ()
      {
      }

      template <typename C>
      dynamic_type<C>::
      dynamic_type (const std::basic_string<C>& type)
          : type_ (type)
      {
      }
    }
  }
}
