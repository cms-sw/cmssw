// file      : xsd/cxx/parser/document.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_DOCUMENT_HXX
#define XSD_CXX_PARSER_DOCUMENT_HXX

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
      // If you want to use a different underlying XML parser, all you
      // need to do is to route events to this interface.
      //
      template <typename C>
      class document
      {
      public:
        virtual
        ~document ();

        document (parser_base<C>& root,
                  const std::basic_string<C>& ns,
                  const std::basic_string<C>& name);

      public:
        // The type argument is a type name and namespace from the
        // xsi:type attribute in the form "<name> <namespace>" with
        // the space and namespace part absent if the type does not
        // have a namespace or 0 if xsi:type is not present.
        //
        void
        start_element (const ro_string<C>& ns,
                       const ro_string<C>& name,
                       const ro_string<C>* type);

        void
        end_element (const ro_string<C>& ns, const ro_string<C>& name);

        void
        attribute (const ro_string<C>& ns,
                   const ro_string<C>& name,
                   const ro_string<C>& value);

        void
        characters (const ro_string<C>&);

      protected:
        document ();

        // This function is called to obtain the root element type parser.
        // If the returned pointed is 0 then the whole document content
        // is ignored.
        //
        virtual parser_base<C>*
        start_root_element (const ro_string<C>& ns,
                            const ro_string<C>& name,
                            const ro_string<C>* type);

        // This function is called to indicate the completion of document
        // parsing. The parser argument contains the pointer returned by
        // start_root_element.
        //
        virtual void
        end_root_element (const ro_string<C>& ns,
                          const ro_string<C>& name,
                          parser_base<C>* parser);

      private:
        parser_base<C>* root_;
        std::basic_string<C> ns_;
        std::basic_string<C> name_;
        std::size_t depth_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/document.txx>

#endif  // XSD_CXX_PARSER_DOCUMENT_HXX
