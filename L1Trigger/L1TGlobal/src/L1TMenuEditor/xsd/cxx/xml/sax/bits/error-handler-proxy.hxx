// file      : xsd/cxx/xml/sax/bits/error-handler-proxy.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_SAX_ERROR_HANDLER_PROXY_HXX
#define XSD_CXX_XML_SAX_ERROR_HANDLER_PROXY_HXX

#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/error-handler.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace sax
      {
        namespace bits
        {
          template <typename C>
          class error_handler_proxy: public xercesc::ErrorHandler
          {
          public:
            error_handler_proxy (error_handler<C>& eh)
                : failed_ (false), eh_ (&eh), native_eh_ (0)
            {
            }

            error_handler_proxy (xercesc::ErrorHandler& eh)
                : failed_ (false), eh_ (0), native_eh_ (&eh)
            {
            }

          public:
            virtual void
            warning (const xercesc::SAXParseException& e);

            virtual void
            error (const xercesc::SAXParseException& e);

            virtual void
            fatalError (const xercesc::SAXParseException& e);

          public:
            bool
            failed () const
            {
              return failed_;
            }

            virtual void
            resetErrors()
            {
              failed_ = false;
            }

          private:
            typedef typename error_handler<C>::severity severity;

            void
            handle (const xercesc::SAXParseException&, severity);

          private:
            bool failed_;
            error_handler<C>* eh_;
            xercesc::ErrorHandler* native_eh_;
          };
        }
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/sax/bits/error-handler-proxy.txx>

#endif  // XSD_CXX_XML_SAX_ERROR_HANDLER_PROXY_HXX
