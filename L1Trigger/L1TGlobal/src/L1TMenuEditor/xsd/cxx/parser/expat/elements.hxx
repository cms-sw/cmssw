// file      : xsd/cxx/parser/expat/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_EXPAT_ELEMENTS_HXX
#define XSD_CXX_PARSER_EXPAT_ELEMENTS_HXX

#include <string>
#include <iosfwd>
#include <cstddef> // std::size_t
#include <vector>

#include <expat.h>

// We only support UTF-8 expat for now.
//
#ifdef XML_UNICODE
#error UTF-16 expat (XML_UNICODE defined) is not supported
#endif

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/error-handler.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/exceptions.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/document.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/error-handler.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace expat
      {
        // Simple auto pointer for Expat's XML_Parser object.
        //
        struct parser_auto_ptr
        {
          ~parser_auto_ptr ()
          {
            if (parser_ != 0)
              XML_ParserFree (parser_);
          }

          explicit
          parser_auto_ptr (XML_Parser parser = 0)
              : parser_ (parser)
          {
          }

          parser_auto_ptr&
          operator= (XML_Parser parser)
          {
            if (parser_ != 0)
              XML_ParserFree (parser_);

            parser_ = parser;
            return *this;
          }

        public:
          operator XML_Parser ()
          {
            return parser_;
          }

        private:
          parser_auto_ptr (const parser_auto_ptr&);

          parser_auto_ptr&
          operator= (const parser_auto_ptr&);

        private:
          XML_Parser parser_;
        };


        //
        //
        template <typename C>
        struct document: cxx::parser::document<C> // VC 7.1 likes it qualified
        {
        public:
          document (parser_base<C>&,
                    const C* root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>&,
                    const std::basic_string<C>& root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>&,
                    const C* root_element_namespace,
                    const C* root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>&,
                    const std::basic_string<C>& root_element_namespace,
                    const std::basic_string<C>& root_element_name,
                    bool polymorphic = false);

        protected:
          document (bool polymorphic = false);

        public:
          // Parse a local file. The file is accessed with std::ifstream
          // in binary mode. The std::ios_base::failure exception is used
          // to report io errors (badbit and failbit).
          void
          parse (const std::basic_string<C>& file);

          // Parse a local file with a user-provided error_handler
          // object. The file is accessed with std::ifstream in binary
          // mode. The std::ios_base::failure exception is used to report
          // io errors (badbit and failbit).
          //
          void
          parse (const std::basic_string<C>& file, xml::error_handler<C>&);

        public:
          // System id is a "system" identifier of the resources (e.g.,
          // URI or a full file path). Public id is a "public" identifier
          // of the resource (e.g., application-specific name or relative
          // file path). System id is used to resolve relative paths.
          // In diagnostics messages system id is used if public id is
          // not available. Otherwise public id is used.
          //

          // Parse std::istream.
          //
          void
          parse (std::istream&);

          // Parse std::istream with a user-provided error_handler object.
          //
          void
          parse (std::istream&, xml::error_handler<C>&);

          // Parse std::istream with a system id.
          //
          void
          parse (std::istream&, const std::basic_string<C>& system_id);

          // Parse std::istream with a system id and a user-provided
          // error_handler object.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 xml::error_handler<C>&);

          // Parse std::istream with system and public ids.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id);

          // Parse std::istream with system and public ids and a user-provided
          // error_handler object.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 xml::error_handler<C>&);

        public:
          // Parse a chunk of input. You can call these functions multiple
          // times with the last call having the last argument true.
          //
          void
          parse (const void* data, std::size_t size, bool last);

          void
          parse (const void* data, std::size_t size, bool last,
                 xml::error_handler<C>&);

          void
          parse (const void* data, std::size_t size, bool last,
                 const std::basic_string<C>& system_id);

          void
          parse (const void* data, std::size_t size, bool last,
                 const std::basic_string<C>& system_id,
                 xml::error_handler<C>&);

          void
          parse (const void* data, std::size_t size, bool last,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id);

          void
          parse (const void* data, std::size_t size, bool last,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 xml::error_handler<C>&);

        public:
          // Low-level Expat-specific parsing API. A typical use case
          // would look like this (pseudo-code):
          //
          // xxx_pimpl root;
          // document doc (root, "root");
          //
          // root.pre ();
          // doc.parse_begin (xml_parser, "file.xml");
          //
          // while (more_stuff_to_parse &&)
          // {
          //    // Call XML_Parse or XML_ParseBuffer.
          //
          //    if (status == XML_STATUS_ERROR)
          //      break;
          // }
          //
          // // Call parse_end even in case of an error to translate
          // // XML and Schema errors to exceptions or error_handler
          // // calls.
          // //
          // doc.parse_end ();
          // result_type result (root.post_xxx ());
          //
          // Notes:
          //
          // 1. If your XML instances use XML namespaces, the
          //    XML_ParserCreateNS functions should be used to create the
          //    XML parser. Space (XML_Char (' ')) should be used as a
          //    separator (the second argument to XML_ParserCreateNS).
          //
          void
          parse_begin (XML_Parser);

          void
          parse_begin (XML_Parser, const std::basic_string<C>& public_id);

          void
          parse_begin (XML_Parser, xml::error_handler<C>&);

          void
          parse_begin (XML_Parser,
                       const std::basic_string<C>& public_id,
                       xml::error_handler<C>&);
          void
          parse_end ();

          // Event routing.
          //
        public:
          static void XMLCALL
          start_element_thunk_ (void*, const XML_Char*, const XML_Char**);

          static void XMLCALL
          end_element_thunk_ (void*, const XML_Char*);

          static void XMLCALL
          characters_thunk_ (void*, const XML_Char*, int);

          static void XMLCALL
          start_namespace_decl_thunk_ (
            void*, const XML_Char*, const XML_Char*);

          static void XMLCALL
          end_namespace_decl_thunk_ (void*, const XML_Char*);

        protected:
          void
          start_element_ (const XML_Char* ns_name, const XML_Char** atts);

          void
          end_element_ (const XML_Char* ns_name);

          void
          characters_ (const XML_Char* s, std::size_t n);

          void
          start_namespace_decl_ (const XML_Char* prefix, const XML_Char* ns);

          void
          end_namespace_decl_ (const XML_Char* prefix);

        protected:
          void
          set ();

          void
          clear ();

          bool
          parse (std::istream&,
                 const std::basic_string<C>* system_id,
                 const std::basic_string<C>* public_id,
                 xml::error_handler<C>&);

          bool
          parse (const void* data, std::size_t size, bool last,
                 const std::basic_string<C>* system_id,
                 const std::basic_string<C>* public_id,
                 xml::error_handler<C>&);


          void
          translate_schema_exception (const schema_exception<C>& e);

        protected:
          XML_Parser xml_parser_;
          parser_auto_ptr auto_xml_parser_;

          xml::error_handler<C>* eh_;
          error_handler<C> default_eh_;
          std::basic_string<C> public_id_;

          bool polymorphic_;

          // Namespace-prefix mapping. Only maintained in the polymorphic
          // case.
          //
          struct ns_decl
          {
            ns_decl (const std::basic_string<C>& p,
                     const std::basic_string<C>& n)
                : prefix (p), ns (n)
            {
            }

            std::basic_string<C> prefix;
            std::basic_string<C> ns;
          };

          typedef std::vector<ns_decl> ns_decls;

          ns_decls ns_decls_;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/expat/elements.txx>

#endif  // XSD_CXX_PARSER_EXPAT_ELEMENTS_HXX
