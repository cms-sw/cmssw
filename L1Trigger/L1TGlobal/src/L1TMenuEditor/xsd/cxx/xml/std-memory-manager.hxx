// file      : xsd/cxx/xml/std-memory-manager.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_STD_MEMORY_MANAGER_HXX
#define XSD_CXX_XML_STD_MEMORY_MANAGER_HXX

#include <new> // operator new, delete
#include <xercesc/framework/MemoryManager.hpp>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      class std_memory_manager: public xercesc::MemoryManager
      {
      public:
        virtual void*
#if _XERCES_VERSION >= 30000
        allocate(XMLSize_t size)
#else
        allocate(size_t size)
#endif
        {
          return operator new (size);
        }

        virtual void
        deallocate(void* p)
        {
          if (p)
	    operator delete (p);
        }

#if _XERCES_VERSION >= 30000
        virtual xercesc::MemoryManager*
        getExceptionMemoryManager()
        {
          return xercesc::XMLPlatformUtils::fgMemoryManager;
        }
#endif
      };
    }
  }
}

#endif  // XSD_CXX_XML_STD_MEMORY_MANAGER_HXX
