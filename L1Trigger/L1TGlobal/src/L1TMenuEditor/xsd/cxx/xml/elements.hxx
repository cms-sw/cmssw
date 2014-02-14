// file      : xsd/cxx/xml/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_ELEMENTS_HXX
#define XSD_CXX_XML_ELEMENTS_HXX

#include <string>

#include <xercesc/util/PlatformUtils.hpp>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <typename C>
      class properties
      {
      public:
        struct argument {};


        // Schema location properties. Note that all locations are
        // relative to an instance document unless they are full
        // URIs. For example if you want to use a local schema then
        // you will need to use 'file:///absolute/path/to/your/schema'.
        //

        // Add a location for a schema with a target namespace.
        //
        void
        schema_location (const std::basic_string<C>& namespace_,
                         const std::basic_string<C>& location);

        // Add a location for a schema without a target namespace.
        //
        void
        no_namespace_schema_location (const std::basic_string<C>& location);

      public:
        const std::basic_string<C>&
        schema_location () const
        {
          return schema_location_;
        }

        const std::basic_string<C>&
        no_namespace_schema_location () const
        {
          return no_namespace_schema_location_;
        }

      private:
        std::basic_string<C> schema_location_;
        std::basic_string<C> no_namespace_schema_location_;
      };


      //
      //

      template <typename C>
      std::basic_string<C>
      prefix (const std::basic_string<C>& n);

      template <typename C>
      std::basic_string<C>
      uq_name (const std::basic_string<C>& n);


      //
      //

      inline void
      initialize ()
      {
        xercesc::XMLPlatformUtils::Initialize ();
      }

      inline void
      terminate ()
      {
        xercesc::XMLPlatformUtils::Terminate ();
      }

      struct auto_initializer
      {
        auto_initializer (bool initialize = true, bool terminate = true)
            : terminate_ (initialize && terminate)
        {
          if (initialize)
            xml::initialize ();
        }

        ~auto_initializer ()
        {
          if (terminate_)
            terminate ();
        }

      private:
        bool terminate_;
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/elements.txx>

#endif  // XSD_CXX_XML_ELEMENTS_HXX
