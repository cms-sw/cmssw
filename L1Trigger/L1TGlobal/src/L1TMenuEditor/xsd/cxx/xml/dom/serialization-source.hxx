// file      : xsd/cxx/xml/dom/serialization-source.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_DOM_SERIALIZATION_SOURCE_HXX
#define XSD_CXX_XML_DOM_SERIALIZATION_SOURCE_HXX

#include <string>
#include <cstring> // std::memcpy
#include <ostream>

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>
#include <xercesc/framework/XMLFormatter.hpp> // XMLFormatTarget, XMLFormatter

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/error-handler.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/auto-ptr.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/elements.hxx> // name
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-header.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      namespace dom
      {
        //
        //
        template <typename C>
        xercesc::DOMAttr&
        create_attribute (const C* name, xercesc::DOMElement&);

        template <typename C>
        xercesc::DOMAttr&
        create_attribute (const C* name, const C* ns, xercesc::DOMElement&);

        template <typename C>
        xercesc::DOMElement&
        create_element (const C* name, xercesc::DOMElement&);

        template <typename C>
        xercesc::DOMElement&
        create_element (const C* name, const C* ns, xercesc::DOMElement&);

        // Serialization flags.
        //
        const unsigned long no_xml_declaration = 0x00010000UL;
        const unsigned long dont_pretty_print  = 0x00020000UL;

        template <typename C>
        xml::dom::auto_ptr<xercesc::DOMDocument>
        serialize (const std::basic_string<C>& root_element,
                   const std::basic_string<C>& root_element_namespace,
                   const namespace_infomap<C>& map,
                   unsigned long flags);

        // This one helps Sun C++ to overcome its fears.
        //
        template <typename C>
        inline xml::dom::auto_ptr<xercesc::DOMDocument>
        serialize (const C* root_element,
                   const C* root_element_namespace,
                   const namespace_infomap<C>& map,
                   unsigned long flags)
        {
          return serialize (std::basic_string<C> (root_element),
                            std::basic_string<C> (root_element_namespace),
                            map,
                            flags);
        }

        //
        //
        template <typename C>
        bool
        serialize (xercesc::XMLFormatTarget& target,
                   const xercesc::DOMDocument& doc,
                   const std::basic_string<C>& enconding,
                   error_handler<C>& eh,
                   unsigned long flags);

        template <typename C>
        bool
        serialize (xercesc::XMLFormatTarget& target,
                   const xercesc::DOMDocument& doc,
                   const std::basic_string<C>& enconding,
                   xercesc::DOMErrorHandler& eh,
                   unsigned long flags);


        class ostream_format_target: public xercesc::XMLFormatTarget
        {
        public:
          ostream_format_target (std::ostream& os)
              : n_ (0), os_ (os)
          {
          }

        public:
          // I know, some of those consts are stupid. But that's what
          // Xerces folks put into their interfaces and VC-7.1 thinks
          // there are different signatures if one strips this fluff off.
          //
          virtual void
          writeChars (const XMLByte* const buf,
#if _XERCES_VERSION >= 30000
                      const XMLSize_t size,
#else
                      const unsigned int size,
#endif
                      xercesc::XMLFormatter* const)
          {
            // Ignore the write request if there was a stream failure and the
            // stream is not using exceptions.
            //
            if (os_.fail ())
              return;

            // Flush the buffer if the block is too large or if we don't have
            // any space left.
            //
            if ((size >= buf_size_ / 8 || n_ + size > buf_size_) && n_ != 0)
            {
              os_.write (buf_, static_cast<std::streamsize> (n_));
              n_ = 0;

              if (os_.fail ())
                return;
            }

            if (size < buf_size_ / 8)
            {
              std::memcpy (buf_ + n_, reinterpret_cast<const char*> (buf), size);
              n_ += size;
            }
            else
              os_.write (reinterpret_cast<const char*> (buf),
                         static_cast<std::streamsize> (size));
          }


          virtual void
          flush ()
          {
            // Ignore the flush request if there was a stream failure
            // and the stream is not using exceptions.
            //
            if (!os_.fail ())
            {
              if (n_ != 0)
              {
                os_.write (buf_, static_cast<std::streamsize> (n_));
                n_ = 0;

                if (os_.fail ())
                  return;
              }

              os_.flush ();
            }
          }

        private:
          static const std::size_t buf_size_ = 1024;
          char buf_[buf_size_];
          std::size_t n_;
          std::ostream& os_;
        };
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/serialization-source.txx>

#endif  // XSD_CXX_XML_DOM_SERIALIZATION_SOURCE_HXX
