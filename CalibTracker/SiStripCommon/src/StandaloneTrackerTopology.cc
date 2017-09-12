#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h"

#include <memory>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include "FWCore/Utilities/interface/Exception.h"

using namespace xercesc;

namespace {
  // copy into a std::string and clean up the C-style string
  inline std::string xmlc_to_stdstring(char* cstr)
  {
    std::string str{cstr};
    XMLString::release(&cstr);
    return str;
  }
  // transcode, convert into a std::string and clean up the intermediate C-style string
  inline std::string xmlc_to_stdstring(const XMLCh* xcstr)
  {
    char* cstr = XMLString::transcode(xcstr);
    std::string str{cstr};
    XMLString::release(&cstr);
    return str;
  }

  // RAII release the XMLCh*
  class auto_XMLString
  {
  public:
    auto_XMLString(XMLCh* xch) : m_xch(xch) {}
    XMLCh* get() const { return m_xch; }
    ~auto_XMLString() { if ( m_xch ) { XMLString::release(&m_xch); } }
    // avoid double release: make this class move-only
    auto_XMLString(const auto_XMLString&) = delete;
    auto_XMLString& operator=(const auto_XMLString&) = delete;
    auto_XMLString(auto_XMLString&& other) { m_xch = other.m_xch; other.m_xch = nullptr; }
    auto_XMLString& operator=(auto_XMLString&& other) { m_xch = other.m_xch; other.m_xch = nullptr; return *this; }
  private:
    XMLCh* m_xch;
  };

  std::string getAttr( const DOMNamedNodeMap* attrMap, const auto_XMLString& attrName )
  {
    if ( attrMap->getNamedItem(attrName.get()) ) {
      return xmlc_to_stdstring(attrMap->getNamedItem(attrName.get())->getTextContent());
    } else {
      return std::string{""};
    }
  }

  // split into tokens and convert them to uint32_t
  inline std::vector<uint32_t> split_string_to_uints(const std::string& str)
  {
    std::vector<uint32_t> out{};
    std::size_t iStart{str.find_first_not_of(" ,\n")}, iEnd{};
    while ( std::string::npos != iStart ) {
      iEnd = str.find_first_of(" ,\n", iStart);
      out.push_back(std::stoul(str.substr(iStart, iEnd), nullptr, 0));
      iStart = str.find_first_not_of(" ,\n", iEnd);
    }
    return out;
  }
}

namespace StandaloneTrackerTopology {
TrackerTopology fromTrackerParametersXML( const std::string& xmlFileName )
{
  TrackerTopology::PixelBarrelValues pxbVals;
  TrackerTopology::PixelEndcapValues pxfVals;
  TrackerTopology::TIBValues tibVals;
  TrackerTopology::TIDValues tidVals;
  TrackerTopology::TOBValues tobVals;
  TrackerTopology::TECValues tecVals;

  try {
    cms::concurrency::xercesInitialize();
  } catch ( const XMLException& xmlEx ) {
    throw cms::Exception("StandaloneTrackerTopology",
        "XML exception at initialization : " + xmlc_to_stdstring(xmlEx.getMessage()));
  }

  { // scope for the parser, and all dependent DOM manipulation
  std::unique_ptr<XercesDOMParser> parser{new XercesDOMParser()};
  parser->setValidationScheme(XercesDOMParser::Val_Always);

  std::unique_ptr<ErrorHandler> errHandler{static_cast<ErrorHandler*>(new HandlerBase())};
  parser->setErrorHandler(errHandler.get());

  try {
    parser->parse(xmlFileName.c_str());
  } catch ( const XMLException& xmlEx ) {
    throw cms::Exception("StandaloneTrackerTopology",
        "XML exception when parsing " + xmlFileName + " : " + xmlc_to_stdstring(xmlEx.getMessage()));
  } catch ( const DOMException& domEx ) {
    throw cms::Exception("StandaloneTrackerTopology",
        "DOM exception when parsing " + xmlFileName + " : " + xmlc_to_stdstring(domEx.getMessage()));
  } catch ( const SAXException& saxEx ) {
    throw cms::Exception("StandaloneTrackerTopology",
        "SAX exception when parsing " + xmlFileName + " : " + xmlc_to_stdstring(saxEx.getMessage()));
  }

  const std::string subdetName{"Subdetector"};
  auto_XMLString nm_type{XMLString::transcode("type")};
  auto_XMLString nm_name{XMLString::transcode("name")};
  auto_XMLString nm_nEntries{XMLString::transcode("nEntries")};

  try {
    DOMDocument* doc{parser->getDocument()};
    DOMElement* docRootNode{doc->getDocumentElement()};
    DOMNodeIterator* walker = doc->createNodeIterator(docRootNode, DOMNodeFilter::SHOW_ELEMENT, nullptr, true);
    for ( DOMNode* currentNode = walker->nextNode(); currentNode; currentNode = walker->nextNode() ) {
      const auto thisNodeName = xmlc_to_stdstring(currentNode->getNodeName());
      if ( thisNodeName == "Vector" ) {
        const auto attrs = currentNode->getAttributes();
        const auto att_type = getAttr(attrs, nm_type);
        if ( att_type == "numeric" ) {
          const auto att_name = getAttr(attrs, nm_name);;
          if ( 0 == att_name.compare(0, subdetName.size(), subdetName) ) {
            const auto att_nEntries = getAttr(attrs, nm_nEntries);
            const std::size_t nEntries = att_nEntries.empty() ? 0 : std::stoul(att_nEntries);
            const auto vals = split_string_to_uints(xmlc_to_stdstring(currentNode->getTextContent()));
            if ( nEntries != vals.size() ) {
              throw cms::Exception("StandaloneTrackerTopology",
                  ("Problem parsing element with name '"+att_name+"' from '"+xmlFileName+"': "+
                   "'nEntries' attribute claims "+std::to_string(nEntries)+" elements, but parsed "+std::to_string(vals.size())));
            }
            const auto subDet = std::stoi(att_name.substr(subdetName.size()));
            switch (subDet) {
              case PixelSubdetector::PixelBarrel: // layer, ladder module
                pxbVals.layerStartBit_        = vals[0];
                pxbVals.ladderStartBit_       = vals[1];
                pxbVals.moduleStartBit_       = vals[2];

                pxbVals.layerMask_            = vals[3];
                pxbVals.ladderMask_           = vals[4];
                pxbVals.moduleMask_           = vals[5];
                break;

              case PixelSubdetector::PixelEndcap: // side, disk, blade, panel, module
                pxfVals.sideStartBit_         = vals[0];
                pxfVals.diskStartBit_         = vals[1];
                pxfVals.bladeStartBit_        = vals[2];
                pxfVals.panelStartBit_        = vals[3];
                pxfVals.moduleStartBit_       = vals[4];

                pxfVals.sideMask_             = vals[5];
                pxfVals.diskMask_             = vals[6];
                pxfVals.bladeMask_            = vals[7];
                pxfVals.panelMask_            = vals[8];
                pxfVals.moduleMask_           = vals[9];
                break;

              case StripSubdetector::TIB: // layer, str_fw_bw, str_int_ext, str, module, ster
                tibVals.layerStartBit_        = vals[ 0];
                tibVals.str_fw_bwStartBit_    = vals[ 1];
                tibVals.str_int_extStartBit_  = vals[ 2];
                tibVals.strStartBit_          = vals[ 3];
                tibVals.moduleStartBit_       = vals[ 4];
                tibVals.sterStartBit_         = vals[ 5];

                tibVals.layerMask_            = vals[ 6];
                tibVals.str_fw_bwMask_        = vals[ 7];
                tibVals.str_int_extMask_      = vals[ 8];
                tibVals.strMask_              = vals[ 9];
                tibVals.moduleMask_           = vals[10];
                tibVals.sterMask_             = vals[11];
                break;

              case StripSubdetector::TID: // side, wheel, ring, module_fw_bw, module, ster
                tidVals.sideStartBit_         = vals[ 0];
                tidVals.wheelStartBit_        = vals[ 1];
                tidVals.ringStartBit_         = vals[ 2];
                tidVals.module_fw_bwStartBit_ = vals[ 3];
                tidVals.moduleStartBit_       = vals[ 4];
                tidVals.sterStartBit_         = vals[ 5];

                tidVals.sideMask_             = vals[ 6];
                tidVals.wheelMask_            = vals[ 7];
                tidVals.ringMask_             = vals[ 8];
                tidVals.module_fw_bwMask_     = vals[ 9];
                tidVals.moduleMask_           = vals[10];
                tidVals.sterMask_             = vals[11];
                break;

              case StripSubdetector::TOB: // layer, rod_fw_bw, rod, module, ster
                tobVals.layerStartBit_        = vals[0];
                tobVals.rod_fw_bwStartBit_    = vals[1];
                tobVals.rodStartBit_          = vals[2];
                tobVals.moduleStartBit_       = vals[3];
                tobVals.sterStartBit_         = vals[4];

                tobVals.layerMask_            = vals[5];
                tobVals.rod_fw_bwMask_        = vals[6];
                tobVals.rodMask_              = vals[7];
                tobVals.moduleMask_           = vals[8];
                tobVals.sterMask_             = vals[9];
                break;

              case StripSubdetector::TEC: // side, wheel, petal_fw_bw, petal, ring, module, ster
                tecVals.sideStartBit_        = vals[ 0];
                tecVals.wheelStartBit_       = vals[ 1];
                tecVals.petal_fw_bwStartBit_ = vals[ 2];
                tecVals.petalStartBit_       = vals[ 3];
                tecVals.ringStartBit_        = vals[ 4];
                tecVals.moduleStartBit_      = vals[ 5];
                tecVals.sterStartBit_        = vals[ 6];

                tecVals.sideMask_            = vals[ 7];
                tecVals.wheelMask_           = vals[ 8];
                tecVals.petal_fw_bwMask_     = vals[ 9];
                tecVals.petalMask_           = vals[10];
                tecVals.ringMask_            = vals[11];
                tecVals.moduleMask_          = vals[12];
                tecVals.sterMask_            = vals[13];
                break;
            }
          }
        }
      }
    }
  } catch ( const DOMException& domEx ) {
    throw cms::Exception("StandaloneTrackerTopology",
        "DOM exception in "+xmlFileName+" : "+xmlc_to_stdstring(domEx.getMessage()));
  }

  } // parser and DOM scope
  cms::concurrency::xercesTerminate();

  return TrackerTopology(pxbVals, pxfVals, tecVals, tibVals, tidVals, tobVals);
}
}
