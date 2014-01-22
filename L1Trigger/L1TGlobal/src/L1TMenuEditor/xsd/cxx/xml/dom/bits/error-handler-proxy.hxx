// file      : xsd/cxx/xml/dom/bits/error-handler-proxy.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_BITS_ERROR_HANDLER_PROXY_HXX
#define XSD_CXX_XML_DOM_BITS_ERROR_HANDLER_PROXY_HXX

#include <xercesc/dom/DOMError.hpp>
#include <xercesc/dom/DOMLocator.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/error-handler.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        namespace bits
        {
          template <typename C>
          class error_handler_proxy: public xercesc::DOMErrorHandler
          {
          public:
            error_handler_proxy (error_handler<C>& eh)
                : failed_ (false), eh_ (&eh), native_eh_ (0)
            {
            }

            error_handler_proxy (xercesc::DOMErrorHandler& eh)
              : failed_ (false), eh_ (0), native_eh_ (&eh)
            {
            }

            virtual bool
            handleError (const xercesc::DOMError& e);

            bool
            failed () const
            {
              return failed_;
            }

          private:
            bool failed_;
            error_handler<C>* eh_;
            xercesc::DOMErrorHandler* native_eh_;
          };
        }
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/bits/error-handler-proxy.txx>

#endif  // XSD_CXX_XML_DOM_BITS_ERROR_HANDLER_PROXY_HXX
