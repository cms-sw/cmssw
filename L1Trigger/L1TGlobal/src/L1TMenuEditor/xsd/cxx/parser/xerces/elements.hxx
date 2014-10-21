// file      : xsd/cxx/parser/xerces/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_PARSER_XERCES_ELEMENTS_HXX
#define XSD_CXX_PARSER_XERCES_ELEMENTS_HXX

#include <memory>  // std::auto_ptr
#include <string>
#include <iosfwd>
#include <vector>

#include <xercesc/sax/Locator.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/error-handler.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/exceptions.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/document.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace xerces
      {
        //
        //
        struct flags
        {
          // Use the following flags to modify the default behavior
          // of the parsing functions.
          //

          // Do not try to validate instance documents.
          //
          static const unsigned long dont_validate = 0x00000001;

          // Do not initialize the Xerces-C++ runtime.
          //
          static const unsigned long dont_initialize = 0x00000002;

          // Disable handling of subsequent imports for the same namespace
          // in Xerces-C++ 3.1.0 and later.
          //
          static const unsigned long no_multiple_imports = 0x00000004;

        public:
          flags (unsigned long x = 0)
              : x_ (x)
          {
          }

          operator unsigned long () const
          {
            return x_;
          }

        private:
          unsigned long x_;
        };


        // Parsing properties. Refer to xsd/cxx/xml/elements.hxx for
        // XML-related properties.
        //
        template <typename C>
        class properties: public xml::properties<C>
        {
        };

        //
        //
        template <typename C>
        struct document: cxx::parser::document<C> // VC 7.1 likes it qualified
        {
        public:
          document (parser_base<C>& root,
                    const C* root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>& root,
                    const std::basic_string<C>& root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>& root,
                    const C* root_element_namespace,
                    const C* root_element_name,
                    bool polymorphic = false);

          document (parser_base<C>& root,
                    const std::basic_string<C>& root_element_namespace,
                    const std::basic_string<C>& root_element_name,
                    bool polymorphic = false);

        protected:
          explicit
          document (bool polymorphic = false);

        public:
          // Parse URI or a local file. We have to overload it for const C*
          // bacause xercesc::InputSource has an implicit constructor that
          // takes const char*.
          //
          void
          parse (const std::basic_string<C>& uri,
                 flags = 0,
                 const properties<C>& = properties<C> ());

          void
          parse (const C* uri,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse URI or a local file with a user-provided error_handler
          // object.
          //
          void
          parse (const std::basic_string<C>& uri,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());

          void
          parse (const C* uri,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse URI or a local file with a user-provided ErrorHandler
          // object. Note that you must initialize the Xerces-C++ runtime
          // before calling these functions.
          //
          void
          parse (const std::basic_string<C>& uri,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());

          void
          parse (const C* uri,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse URI or a local file using a user-provided SAX2XMLReader
          // object. Note that you must initialize the Xerces-C++ runtime
          // before calling these functions.
          //
          void
          parse (const std::basic_string<C>& uri,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());

          void
          parse (const C* uri,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


        public:
          // System id is a "system" identifier of the resources (e.g.,
          // URI or a full file path). Public id is a "public" identifier
          // of the resource (e.g., an application-specific name or a
          // relative file path). System id is used to resolve relative
          // paths. In diagnostics messages system id is used if public
          // id is not available. Otherwise public id is used.
          //

          // Parse std::istream.
          //
          void
          parse (std::istream&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with a user-provided error_handler object.
          //
          void
          parse (std::istream&,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with a user-provided ErrorHandler object.
          // Note that you must initialize the Xerces-C++ runtime before
          // calling this function.
          //
          void
          parse (std::istream&,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream using a user-provided SAX2XMLReader object.
          // Note that you must initialize the Xerces-C++ runtime before
          // calling this function.
          //
          void
          parse (std::istream&,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


        public:
          // Parse std::istream with a system id.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with a system id and a user-provided
          // error_handler object.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with a system id and a user-provided
          // ErrorHandler object. Note that you must initialize the
          // Xerces-C++ runtime before calling this function.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with a system id using a user-provided
          // SAX2XMLReader object. Note that you must initialize the
          // Xerces-C++ runtime before calling this function.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());



        public:
          // Parse std::istream with system and public ids.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with system and public ids and a user-provided
          // error_handler object.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with system and public ids and a user-provided
          // ErrorHandler object. Note that you must initialize the Xerces-C++
          // runtime before calling this function.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse std::istream with system and public ids using a user-
          // provided SAX2XMLReader object. Note that you must initialize
          // the Xerces-C++ runtime before calling this function.
          //
          void
          parse (std::istream&,
                 const std::basic_string<C>& system_id,
                 const std::basic_string<C>& public_id,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


        public:
          // Parse InputSource. Note that you must initialize the Xerces-C++
          // runtime before calling this function.
          //
          void
          parse (const xercesc::InputSource&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse InputSource with a user-provided error_handler object.
          // Note that you must initialize the Xerces-C++ runtime before
          // calling this function.
          //
          void
          parse (const xercesc::InputSource&,
                 xml::error_handler<C>&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse InputSource with a user-provided ErrorHandler object.
          // Note that you must initialize the Xerces-C++ runtime before
          // calling this function.
          //
          void
          parse (const xercesc::InputSource&,
                 xercesc::ErrorHandler&,
                 flags = 0,
                 const properties<C>& = properties<C> ());


          // Parse InputSource using a user-provided SAX2XMLReader object.
          // Note that you must initialize the Xerces-C++ runtime before
          // calling this function.
          //
          void
          parse (const xercesc::InputSource&,
                 xercesc::SAX2XMLReader&,
                 flags = 0,
                 const properties<C>& = properties<C> ());

        private:
          void
          parse (const std::basic_string<C>& uri,
                 xercesc::ErrorHandler&,
                 xercesc::SAX2XMLReader&,
                 flags,
                 const properties<C>&);

          void
          parse (const xercesc::InputSource&,
                 xercesc::ErrorHandler&,
                 xercesc::SAX2XMLReader&,
                 flags,
                 const properties<C>&);

        private:
          std::auto_ptr<xercesc::SAX2XMLReader>
          create_sax_ (flags, const properties<C>&);

        private:
          bool polymorphic_;
        };

        //
        //
        template <typename C>
        struct event_router: xercesc::DefaultHandler
        {
          event_router (cxx::parser::document<C>&, bool polymorphic);

          // I know, some of those consts are stupid. But that's what
          // Xerces folks put into their interfaces and VC 7.1 thinks
          // there are different signatures if one strips this fluff off.
          //
          virtual void
          setDocumentLocator (const xercesc::Locator* const);

          virtual void
          startElement (const XMLCh* const uri,
                        const XMLCh* const lname,
                        const XMLCh* const qname,
                        const xercesc::Attributes& attributes);

          virtual void
          endElement (const XMLCh* const uri,
                      const XMLCh* const lname,
                      const XMLCh* const qname);

#if _XERCES_VERSION >= 30000
          virtual void
          characters (const XMLCh* const s, const XMLSize_t length);
#else
          virtual void
          characters (const XMLCh* const s, const unsigned int length);
#endif

          virtual void
          startPrefixMapping (const XMLCh* const prefix,
                              const XMLCh* const uri);

          virtual void
          endPrefixMapping (const XMLCh* const prefix);

        private:
          void
          set_location (schema_exception<C>&);

        private:
          const xercesc::Locator* loc_;
          cxx::parser::document<C>& consumer_;
          bool polymorphic_;

          // Last element name cache.
          //
          bool last_valid_;
          std::basic_string<C> last_ns_;
          std::basic_string<C> last_name_;

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

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/xerces/elements.txx>

#endif  // XSD_CXX_PARSER_XERCES_ELEMENTS_HXX
