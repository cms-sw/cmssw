// file      : xsd/cxx/xml/dom/bits/error-handler-proxy.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>

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
          bool error_handler_proxy<C>::
          handleError (const xercesc::DOMError& e)
          {
            using xercesc::DOMError;

            if (e.getSeverity() != DOMError::DOM_SEVERITY_WARNING)
              failed_ = true;

            if (native_eh_)
              return native_eh_->handleError (e);
            else
            {
              typedef typename error_handler<C>::severity severity;

              severity s (severity::error);

              switch (e.getSeverity())
              {
              case DOMError::DOM_SEVERITY_WARNING:
                {
                  s = severity::warning;
                  break;
                }
              case DOMError::DOM_SEVERITY_ERROR:
                {
                  s = severity::error;
                  break;
                }
              case DOMError::DOM_SEVERITY_FATAL_ERROR:
                {
                  s = severity::fatal;
                  break;
                }
              }

              xercesc::DOMLocator* loc (e.getLocation ());

#if _XERCES_VERSION >= 30000
              return eh_->handle (
                transcode<C> (loc->getURI ()),
                static_cast<unsigned long> (loc->getLineNumber ()),
                static_cast<unsigned long> (loc->getColumnNumber ()),
                s,
                transcode<C> (e.getMessage ()));
#else
              XMLSSize_t l (loc->getLineNumber ());
              XMLSSize_t c (loc->getColumnNumber ());

              return eh_->handle (
                transcode<C> (loc->getURI ()),
                (l == -1 ? 0 : static_cast<unsigned long> (l)),
                (c == -1 ? 0 : static_cast<unsigned long> (c)),
                s,
                transcode<C> (e.getMessage ()));
#endif
            }
          }
        }
      }
    }
  }
}
