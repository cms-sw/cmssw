// file      : xsd/cxx/tree/exceptions.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // error
      //
      template <typename C>
      error<C>::
      error (tree::severity s,
             const std::basic_string<C>& id,
             unsigned long line,
             unsigned long column,
             const std::basic_string<C>& message)
          : severity_ (s),
            id_ (id),
            line_ (line),
            column_ (column),
            message_ (message)
      {
      }

      template <typename C>
      error<C>::
      error ()
          : severity_ (tree::severity::error), line_ (0), column_ (0)
      {
      }

      template <typename C>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const error<C>& e)
      {
        return os << e.id () << C (':') << e.line () << C (':') << e.column ()
                  << (e.severity () == severity::error
                      ? bits::ex_error_error<C> ()
                      : bits::ex_error_warning<C> ()) << e.message ();
      }

      // diagnostics
      //
      template <typename C>
      std::basic_ostream<C>&
      operator<< (std::basic_ostream<C>& os, const diagnostics<C>& d)
      {
        for (typename diagnostics<C>::const_iterator b (d.begin ()), i (b);
             i != d.end ();
             ++i)
        {
          if (i != b)
            os << C ('\n');

          os << *i;
        }

        return os;
      }

      // parsing
      //
      template <typename C>
      parsing<C>::
      ~parsing () throw ()
      {
      }

      template <typename C>
      parsing<C>::
      parsing ()
      {
      }

      template <typename C>
      parsing<C>::
      parsing (const tree::diagnostics<C>& diagnostics)
          : diagnostics_ (diagnostics)
      {
      }

      template <typename C>
      const char* parsing<C>::
      what () const throw ()
      {
        return "instance document parsing failed";
      }

      template <typename C>
      void parsing<C>::
      print (std::basic_ostream<C>& os) const
      {
        if (diagnostics_.empty ())
          os << bits::ex_parsing_msg<C> ();
        else
          os << diagnostics_;
      }

      // expected_element
      //
      template <typename C>
      expected_element<C>::
      ~expected_element () throw ()
      {
      }

      template <typename C>
      expected_element<C>::
      expected_element (const std::basic_string<C>& name,
                        const std::basic_string<C>& namespace_)
          : name_ (name), namespace__ (namespace_)
      {
      }

      template <typename C>
      const char* expected_element<C>::
      what () const throw ()
      {
        return "expected element not encountered";
      }

      template <typename C>
      void expected_element<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_eel_expected<C> ();

        if (!namespace_ ().empty ())
          os << namespace_ () << C ('#');

        os << name () << C ('\'');
      }

      // unexpected_element
      //
      template <typename C>
      unexpected_element<C>::
      ~unexpected_element () throw ()
      {
      }

      template <typename C>
      unexpected_element<C>::
      unexpected_element (const std::basic_string<C>& encountered_name,
                          const std::basic_string<C>& encountered_namespace,
                          const std::basic_string<C>& expected_name,
                          const std::basic_string<C>& expected_namespace)
          : encountered_name_ (encountered_name),
            encountered_namespace_ (encountered_namespace),
            expected_name_ (expected_name),
            expected_namespace_ (expected_namespace)
      {
      }

      template <typename C>
      const char* unexpected_element<C>::
      what () const throw ()
      {
        return "unexpected element encountered";
      }

      template <typename C>
      void unexpected_element<C>::
      print (std::basic_ostream<C>& os) const
      {
        if (!expected_name ().empty ())
        {
          os << bits::ex_uel_expected<C> ();

          if (!expected_namespace ().empty ())
            os << expected_namespace () << C ('#');

          os << expected_name () << bits::ex_uel_instead<C> ();

          if (!encountered_namespace ().empty ())
            os << encountered_namespace () << C ('#');

          os << encountered_name () << C ('\'');
        }
        else
        {
          os << bits::ex_uel_unexpected<C> ();

          if (!encountered_namespace ().empty ())
            os << encountered_namespace () << C ('#');

          os << encountered_name () << C ('\'');
        }
      }

      // expected_attribute
      //
      template <typename C>
      expected_attribute<C>::
      ~expected_attribute () throw ()
      {
      }

      template <typename C>
      expected_attribute<C>::
      expected_attribute (const std::basic_string<C>& name,
                          const std::basic_string<C>& namespace_)
          : name_ (name), namespace__ (namespace_)
      {
      }

      template <typename C>
      const char* expected_attribute<C>::
      what () const throw ()
      {
        return "expected attribute not encountered";
      }

      template <typename C>
      void expected_attribute<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_eat_expected<C> ();

        if (!namespace_ ().empty ())
          os << namespace_ () << C ('#');

        os << name () << C ('\'');
      }

      // unexpected_enumerator
      //
      template <typename C>
      unexpected_enumerator<C>::
      ~unexpected_enumerator () throw ()
      {
      }

      template <typename C>
      unexpected_enumerator<C>::
      unexpected_enumerator (const std::basic_string<C>& enumerator)
          : enumerator_ (enumerator)
      {
      }

      template <typename C>
      const char* unexpected_enumerator<C>::
      what () const throw ()
      {
        return "unexpected enumerator encountered";
      }

      template <typename C>
      void unexpected_enumerator<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_uen_unexpected<C> () << enumerator () << C ('\'');
      }

      // expected_text_content
      //
      template <typename C>
      const char* expected_text_content<C>::
      what () const throw ()
      {
        return "expected text content";
      }

      template <typename C>
      void expected_text_content<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_etc_msg<C> ();
      }

      // no_type_info
      //
      template <typename C>
      no_type_info<C>::
      ~no_type_info () throw ()
      {
      }

      template <typename C>
      no_type_info<C>::
      no_type_info (const std::basic_string<C>& type_name,
                    const std::basic_string<C>& type_namespace)
          : type_name_ (type_name),
            type_namespace_ (type_namespace)
      {
      }

      template <typename C>
      const char* no_type_info<C>::
      what () const throw ()
      {
        return "no type information available for a type";
      }

      template <typename C>
      void no_type_info<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_nti_no_type_info<C> ();

        if (!type_namespace ().empty ())
          os << type_namespace () << C ('#');

        os << type_name () << C ('\'');
      }

      // no_element_info
      //
      template <typename C>
      no_element_info<C>::
      ~no_element_info () throw ()
      {
      }

      template <typename C>
      no_element_info<C>::
      no_element_info (const std::basic_string<C>& element_name,
                       const std::basic_string<C>& element_namespace)
          : element_name_ (element_name),
            element_namespace_ (element_namespace)
      {
      }

      template <typename C>
      const char* no_element_info<C>::
      what () const throw ()
      {
        return "no parsing or serialization information available for "
          "an element";
      }

      template <typename C>
      void no_element_info<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_nei_no_element_info<C> ();

        if (!element_namespace ().empty ())
          os << element_namespace () << C ('#');

        os << element_name () << C ('\'');
      }

      // not_derived
      //
      template <typename C>
      not_derived<C>::
      ~not_derived () throw ()
      {
      }

      template <typename C>
      not_derived<C>::
      not_derived (const std::basic_string<C>& base_type_name,
                   const std::basic_string<C>& base_type_namespace,
                   const std::basic_string<C>& derived_type_name,
                   const std::basic_string<C>& derived_type_namespace)
          : base_type_name_ (base_type_name),
            base_type_namespace_ (base_type_namespace),
            derived_type_name_ (derived_type_name),
            derived_type_namespace_ (derived_type_namespace)
      {
      }

      template <typename C>
      const char* not_derived<C>::
      what () const throw ()
      {
        return "type is not derived";
      }

      template <typename C>
      void not_derived<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_nd_type<C> ();

        if (!derived_type_namespace ().empty ())
          os << derived_type_namespace () << C ('#');

        os << derived_type_name () << bits::ex_nd_not_derived<C> ();

        if (!base_type_namespace ().empty ())
          os << base_type_namespace () << C ('#');

        os << base_type_name () << C ('\'');
      }

      // duplicate_id
      //
      template <typename C>
      duplicate_id<C>::
      ~duplicate_id () throw ()
      {
      }

      template <typename C>
      duplicate_id<C>::
      duplicate_id (const std::basic_string<C>& id)
          : id_ (id)
      {
      }

      template <typename C>
      const char* duplicate_id<C>::
      what () const throw ()
      {
        return "ID already exist";
      }

      template <typename C>
      void duplicate_id<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_di_id<C> () << id () << bits::ex_di_already_exist<C> ();
      }

      // serialization
      //
      template <typename C>
      serialization<C>::
      ~serialization () throw ()
      {
      }

      template <typename C>
      serialization<C>::
      serialization ()
      {
      }

      template <typename C>
      serialization<C>::
      serialization (const tree::diagnostics<C>& diagnostics)
          : diagnostics_ (diagnostics)
      {
      }

      template <typename C>
      const char* serialization<C>::
      what () const throw ()
      {
        return "serialization failed";
      }

      template <typename C>
      void serialization<C>::
      print (std::basic_ostream<C>& os) const
      {
        if (diagnostics_.empty ())
          os << bits::ex_serialization_msg<C> ();
        else
          os << diagnostics_;
      }


      // no_prefix_mapping
      //
      template <typename C>
      no_prefix_mapping<C>::
      ~no_prefix_mapping () throw ()
      {
      }

      template <typename C>
      no_prefix_mapping<C>::
      no_prefix_mapping (const std::basic_string<C>& prefix)
          : prefix_ (prefix)
      {
      }

      template <typename C>
      const char* no_prefix_mapping<C>::
      what () const throw ()
      {
        return "no mapping provided for a namespace prefix";
      }

      template <typename C>
      void no_prefix_mapping<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_npm_no_mapping<C> () << prefix () << C ('\'');
      }


      // bounds
      //
      template <typename C>
      const char* bounds<C>::
      what () const throw ()
      {
        return "buffer boundary rules have been violated";
      }

      template <typename C>
      void bounds<C>::
      print (std::basic_ostream<C>& os) const
      {
        os << bits::ex_bounds_msg<C> ();
      }
    }
  }
}
