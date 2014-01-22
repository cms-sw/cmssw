// file      : xsd/cxx/tree/bits/literals.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_BITS_LITERALS_HXX
#define XSD_CXX_TREE_BITS_LITERALS_HXX

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      namespace bits
      {
        // Boolean literals
        //
        template<typename C>
        const C*
        true_ ();

        template<typename C>
        const C*
        one ();

        // Float literals: INF -INF NaN.
        //
        template<typename C>
        const C*
        positive_inf ();

        template<typename C>
        const C*
        negative_inf ();

        template<typename C>
        const C*
        nan ();

        // Optional "not present" literal.
        //
        template<typename C>
        const C*
        not_present ();

        // XML Schema namespace
        //
        template <typename C>
        const C*
        xml_schema ();

        // Built-in XML Schema type names.
        //
        template <typename C>
        const C*
        any_type ();

        template <typename C>
        const C*
        any_simple_type ();

        template <typename C>
        const C*
        string ();

        template <typename C>
        const C*
        normalized_string ();

        template <typename C>
        const C*
        token ();

        template <typename C>
        const C*
        name ();

        template <typename C>
        const C*
        nmtoken ();

        template <typename C>
        const C*
        nmtokens ();

        template <typename C>
        const C*
        ncname ();

        template <typename C>
        const C*
        language ();

        template <typename C>
        const C*
        id ();

        template <typename C>
        const C*
        idref ();

        template <typename C>
        const C*
        idrefs ();

        template <typename C>
        const C*
        any_uri ();

        template <typename C>
        const C*
        qname ();

        template <typename C>
        const C*
        base64_binary ();

        template <typename C>
        const C*
        hex_binary ();

        template <typename C>
        const C*
        date ();

        template <typename C>
        const C*
        date_time ();

        template <typename C>
        const C*
        duration ();

        template <typename C>
        const C*
        gday ();

        template <typename C>
        const C*
        gmonth ();

        template <typename C>
        const C*
        gmonth_day ();

        template <typename C>
        const C*
        gyear ();

        template <typename C>
        const C*
        gyear_month ();

        template <typename C>
        const C*
        time ();

        template <typename C>
        const C*
        entity ();

        template <typename C>
        const C*
        entities ();

        // gday ("---") and gmonth ("--") prefixes.
        //
        template <typename C>
        const C*
        gday_prefix ();

        template <typename C>
        const C*
        gmonth_prefix ();

        // Exception and diagnostics string literals.
        //
        template <typename C>
        const C*
        ex_error_error (); // " error: "

        template <typename C>
        const C*
        ex_error_warning (); // " warning: "

        template <typename C>
        const C*
        ex_parsing_msg (); // "instance document parsing failed"

        template <typename C>
        const C*
        ex_eel_expected (); // "expected element '"

        template <typename C>
        const C*
        ex_uel_expected (); // "expected element '"

        template <typename C>
        const C*
        ex_uel_instead (); // "' instead of '"

        template <typename C>
        const C*
        ex_uel_unexpected (); // "unexpected element '"

        template <typename C>
        const C*
        ex_eat_expected (); // "expected attribute '"

        template <typename C>
        const C*
        ex_uen_unexpected (); // "unexpected enumerator '"

        template <typename C>
        const C*
        ex_etc_msg (); // "expected text content"

        template <typename C>
        const C*
        ex_nti_no_type_info (); // "no type information available for type '"

        template <typename C>
        const C*
        ex_nei_no_element_info (); // "no parsing or serialization information
                                   // available for element '"
        template <typename C>
        const C*
        ex_nd_type (); // "type '"

        template <typename C>
        const C*
        ex_nd_not_derived (); // "' is not derived from '"

        template <typename C>
        const C*
        ex_di_id (); // "ID '"

        template <typename C>
        const C*
        ex_di_already_exist (); // "' already exist"

        template <typename C>
        const C*
        ex_serialization_msg (); // "serialization failed"

        template <typename C>
        const C*
        ex_npm_no_mapping (); // "no mapping provided for namespace prefix '"

        template <typename C>
        const C*
        ex_bounds_msg (); // "buffer boundary rules have been violated"
      }
    }
  }
}

#endif  // XSD_CXX_TREE_BITS_LITERALS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.ixx>
