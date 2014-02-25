// file      : xsd/cxx/tree/text.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_TEXT_HXX
#define XSD_CXX_TREE_TEXT_HXX

#include <string>

#include <xercesc/dom/DOMElement.hpp>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      // Throws expected_text_content.
      //
      template <typename C>
      std::basic_string<C>
      text_content (const xercesc::DOMElement&);
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/text.txx>

#endif // XSD_CXX_TREE_TEXT_HXX
