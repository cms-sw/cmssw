// file      : xsd/cxx/parser/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_ELEMENTS_HXX
#define XSD_CXX_PARSER_ELEMENTS_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // pre() and post() are overridable pre/post callbacks, i.e., the
      // derived parser can override them without calling the base version.
      // _pre() and _post() are not overridable pre/post callbacks in the
      // sense that the derived parser may override them but has to call
      // the base version. The call sequence is as shown below:
      //
      // pre ()
      // _pre ()
      // _post ()
      // post ()
      //
      template <typename C>
      class parser_base
      {
      public:
        virtual
        ~parser_base ();

        virtual void
        pre ();

        virtual void
        _pre ();

        // The type argument is a type name and namespace from the
        // xsi:type attribute in the form "<name> <namespace>" with
        // the space and namespace part absent if the type does not
        // have a namespace or 0 if xsi:type is not present.
        //
        virtual void
        _start_element (const ro_string<C>& ns,
                        const ro_string<C>& name,
                        const ro_string<C>* type) = 0;

        virtual void
        _end_element (const ro_string<C>& ns,
                      const ro_string<C>& name) = 0;

        virtual void
        _attribute (const ro_string<C>& ns,
                    const ro_string<C>& name,
                    const ro_string<C>& value) = 0;

        virtual void
        _characters (const ro_string<C>&) = 0;

        virtual void
        _post ();

        // The post() signature varies depending on the parser return
        // type.
        //

        // Implementation callbacks for _pre and _post. The _pre and _post
        // callbacks should never be called directly. Instead, the *_impl
        // versions should be used. By default _pre_impl and _post_impl
        // simply call _pre and _post respectively.
        //
        virtual void
        _pre_impl ();

        virtual void
        _post_impl ();

      public:
        // Dynamic type in the form "<name> <namespace>" with
        // the space and namespace part absent if the type does
        // not have a namespace. Used in polymorphism-aware code.
        //
        virtual const C*
        _dynamic_type () const;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.txx>

#endif  // XSD_CXX_PARSER_ELEMENTS_HXX
