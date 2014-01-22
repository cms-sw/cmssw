// file      : xsd/cxx/tree/text.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <xercesc/dom/DOMText.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      std::basic_string<C>
      text_content (const xercesc::DOMElement& e)
      {
        using xercesc::DOMNode;
        using xercesc::DOMText;

        DOMNode* n (e.getFirstChild ());

        // Fast path.
        //
        if (n != 0 &&
            n->getNodeType () == DOMNode::TEXT_NODE &&
            n->getNextSibling () == 0)
        {
          DOMText* t (static_cast<DOMText*> (n));

          // Berkeley DB XML DOM does not implement getLength().
          //
#ifndef DBXML_DOM
          return xml::transcode<C> (t->getData (), t->getLength ());
#else
	  return xml::transcode<C> (t->getData ());
#endif
        }

        std::basic_string<C> r;

        for (; n != 0; n = n->getNextSibling ())
        {
          switch (n->getNodeType ())
          {
          case DOMNode::TEXT_NODE:
          case DOMNode::CDATA_SECTION_NODE:
            {
              DOMText* t (static_cast<DOMText*> (n));

              // Berkeley DB XML DOM does not implement getLength().
              //
#ifndef DBXML_DOM
              r += xml::transcode<C> (t->getData (), t->getLength ());
#else
	      r += xml::transcode<C> (t->getData ());
#endif
              break;
            }
          case DOMNode::ELEMENT_NODE:
            {
              throw expected_text_content<C> ();
            }
          default:
            break; // ignore
          }
        }

        return r;
      }
    }
  }
}
