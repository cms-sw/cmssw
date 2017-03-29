/****************************************************************************
 *
 *
 * Authors:
 * F.Ferro ferro@ge.infn.it
 * H. Malbouisson malbouis@cern.ch
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelFramePosition.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <memory>
#include <sstream>

//#define DEBUG 1

//----------------------------------------------------------------------------------------------------

using namespace std;

/**
 * \brief Loads CTPPSPixelDAQMapping and CTPPSPixelAnalysisMask from two XML files.
 **/
class CTPPSPixelDAQMappingESSourceXML: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
public:

  static const std::string tagAnalysisMask;
  static const std::string tagRPixPlane;
  static const std::string tagROC;
  static const std::string tagPixel;
/// Common position tags
  static const std::string tagArm;

/// RP XML tags
  static const std::string tagRPStation;
  static const std::string tagRPPot;


  CTPPSPixelDAQMappingESSourceXML(const edm::ParameterSet &);
  ~CTPPSPixelDAQMappingESSourceXML();

  std::unique_ptr<CTPPSPixelAnalysisMask> produceCTPPSPixelAnalysisMask( const CTPPSPixelAnalysisMaskRcd & );
  std::unique_ptr<CTPPSPixelDAQMapping> produceCTPPSPixelDAQMapping( const CTPPSPixelDAQMappingRcd &);

private:
  unsigned int verbosity;

/// label of the CTPPS sub-system
  string subSystemName;

/// the mapping files
  std::vector<std::string> mappingFileNames;

  struct ConfigBlock
  {
  /// validity interval
    edm::EventRange validityRange;

  /// the mapping files
    std::vector<std::string> mappingFileNames;

  /// the mask files
    std::vector<std::string> maskFileNames;
  };

  vector<ConfigBlock> configuration;

/// index of the current block in 'configuration' array
  unsigned int currentBlock;

/// flag whether the 'currentBlock' index is valid
  bool currentBlockValid;

/// enumeration of XML node types
  enum NodeType { nUnknown, nSkip, nTop, nArm, nRPStation, nRPPot, nRPixPlane, nROC, nPixel };

/// whether to parse a mapping of a mask XML
  enum ParseType { pMapping, pMask };

/// parses XML file
  void ParseXML(ParseType, const string &file, const std::unique_ptr<CTPPSPixelDAQMapping>&, const std::unique_ptr<CTPPSPixelAnalysisMask>&);



/// recursive method to extract Pixel-related information from the DOM tree
  void ParseTreePixel(ParseType, xercesc::DOMNode *, NodeType, unsigned int parentID,
		      const std::unique_ptr<CTPPSPixelDAQMapping>&, const std::unique_ptr<CTPPSPixelAnalysisMask>&);

private:
/// adds the path prefix, if needed
  string CompleteFileName(const string &fn);

/// returns true iff the node is of the given name
  bool Test(xercesc::DOMNode *node, const std::string &name)
  {
    return !(name.compare(xercesc::XMLString::transcode(node->getNodeName())));
  }

/// determines node type
  NodeType GetNodeType(xercesc::DOMNode *);

/// returns the content of the node
  string GetNodeContent(xercesc::DOMNode *parent)
  {
    return string(xercesc::XMLString::transcode(parent->getTextContent()));
  }

/// returns the value of the node
  string GetNodeValue(xercesc::DOMNode *node)
  {
    return string(xercesc::XMLString::transcode(node->getNodeValue()));
  }

/// extracts VFAT's DAQ channel from XML attributes
  CTPPSPixelFramePosition ChipFramePosition(xercesc::DOMNode *chipnode);

  void GetPixels(xercesc::DOMNode *n, std::set<std::pair<unsigned char, unsigned char> > &pixels);

  bool PixelNode(NodeType type)
  {
    return ((type == nArm)||(type == nRPStation)||(type == nRPPot)||(type == nRPixPlane)||(type == nROC));
  }


protected:
/// sets infinite validity of this data
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&);
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace xercesc;

const string CTPPSPixelDAQMappingESSourceXML::tagROC="roc";
const string CTPPSPixelDAQMappingESSourceXML::tagPixel="pixel";
const string CTPPSPixelDAQMappingESSourceXML::tagAnalysisMask="analysisMask";

// common XML position tags
const string CTPPSPixelDAQMappingESSourceXML::tagArm = "arm";

// specific RP XML tags
const string CTPPSPixelDAQMappingESSourceXML::tagRPStation = "station";
const string CTPPSPixelDAQMappingESSourceXML::tagRPPot = "rp_detector_set";
const string CTPPSPixelDAQMappingESSourceXML::tagRPixPlane = "rpix_plane";


static const unsigned int offsetROCinDetId = 13;
//static const unsigned int maskROCinDetId = 0x3;

CTPPSPixelDAQMappingESSourceXML::CTPPSPixelDAQMappingESSourceXML(const edm::ParameterSet& conf) :
  verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
  subSystemName(conf.getUntrackedParameter<string>("subSystem")),
  currentBlock(0),
  currentBlockValid(false)
{
  for (const auto it : conf.getParameter<vector<ParameterSet>>("configuration"))
    {
      ConfigBlock b;
      b.validityRange = it.getParameter<EventRange>("validityRange");
      b.mappingFileNames = it.getParameter< vector<string> >("mappingFileNames");
      b.maskFileNames = it.getParameter< vector<string> >("maskFileNames");
      configuration.push_back(b);
    }

  setWhatProduced(this, &CTPPSPixelDAQMappingESSourceXML::produceCTPPSPixelAnalysisMask, es::Label(subSystemName));
  findingRecord<CTPPSPixelAnalysisMaskRcd>();

  setWhatProduced(this, &CTPPSPixelDAQMappingESSourceXML::produceCTPPSPixelDAQMapping, es::Label(subSystemName));
  findingRecord<CTPPSPixelDAQMappingRcd>();


  LogVerbatim("CTPPSPixelDAQMappingESSourceXML") << " Inside  CTPPSPixelDAQMappingESSourceXML";


}

void CTPPSPixelDAQMappingESSourceXML::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
						     const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  LogVerbatim("CTPPSPixelDAQMappingESSourceXML")
    << ">> CTPPSPixelDAQMappingESSourceXML::setIntervalFor(" << key.name() << ")";

  LogVerbatim("CTPPSPixelDAQMappingESSourceXML")
    << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  currentBlockValid = false;
  for (unsigned int idx = 0; idx < configuration.size(); ++idx)
    {
      const auto &bl = configuration[idx];

      EventID startEventID = bl.validityRange.startEventID();
      if (startEventID == EventID(1, 0, 1))
	startEventID = EventID(1, 0, 0);

      if (startEventID <= iosv.eventID() && iosv.eventID() <= bl.validityRange.endEventID())
	{
	  currentBlockValid = true;
	  currentBlock = idx;

	  const IOVSyncValue begin(startEventID);
	  const IOVSyncValue end(bl.validityRange.endEventID());
	  oValidity = ValidityInterval(begin, end);

	  LogVerbatim("CTPPSPixelDAQMappingESSourceXML")
	    << "    block found: index=" << currentBlock
	    << ", interval=(" << startEventID << " - " << bl.validityRange.endEventID() << ")";

	  return;
	}
    }

  if (!currentBlockValid)
    {
      throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::setIntervalFor") <<
	"No configuration for event " << iosv.eventID();
    }
}



CTPPSPixelDAQMappingESSourceXML::~CTPPSPixelDAQMappingESSourceXML()
{
}



string CTPPSPixelDAQMappingESSourceXML::CompleteFileName(const string &fn)
{
  FileInPath fip(fn);
  return fip.fullPath();
}



std::unique_ptr<CTPPSPixelDAQMapping> CTPPSPixelDAQMappingESSourceXML::produceCTPPSPixelDAQMapping( const CTPPSPixelDAQMappingRcd & )
{
  assert(currentBlockValid);

  auto mapping = std::make_unique<CTPPSPixelDAQMapping>();
  auto mask = std::make_unique<CTPPSPixelAnalysisMask>();

// initialize Xerces
  try
    {
      XMLPlatformUtils::Initialize();
    }
  catch (const XMLException& toCatch)
    {
      char* message = XMLString::transcode(toCatch.getMessage());
      throw cms::Exception("CTPPSPixelDAQMappingESSourceXML") << "An XMLException caught with message: " << message << ".\n";
      XMLString::release(&message);
    }

// load mapping files
  for (const auto &fn : configuration[currentBlock].mappingFileNames)
    ParseXML(pMapping, CompleteFileName(fn), mapping, mask);

// load mask files
  for (const auto &fn : configuration[currentBlock].maskFileNames)
    ParseXML(pMask, CompleteFileName(fn), mapping, mask);

// release Xerces
  XMLPlatformUtils::Terminate();

// commit the product
  return mapping;
}

std::unique_ptr<CTPPSPixelAnalysisMask> CTPPSPixelDAQMappingESSourceXML::produceCTPPSPixelAnalysisMask( const CTPPSPixelAnalysisMaskRcd & )
{
  assert(currentBlockValid);

  auto mapping = std::make_unique<CTPPSPixelDAQMapping>();
  auto mask = std::make_unique<CTPPSPixelAnalysisMask>();

// initialize Xerces
  try
    {
      XMLPlatformUtils::Initialize();
    }
  catch (const XMLException& toCatch)
    {
      char* message = XMLString::transcode(toCatch.getMessage());
      throw cms::Exception("CTPPSPixelDAQMappingESSourceXML") << "An XMLException caught with message: " << message << ".\n";
      XMLString::release(&message);
    }

// load mapping files
  for (const auto &fn : configuration[currentBlock].mappingFileNames)
    ParseXML(pMapping, CompleteFileName(fn), mapping, mask);

// load mask files
  for (const auto &fn : configuration[currentBlock].maskFileNames)
    ParseXML(pMask, CompleteFileName(fn), mapping, mask);

// release Xerces
  XMLPlatformUtils::Terminate();

// commit the products
//return edm::es::products(mapping, mask);
  return mask;
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelDAQMappingESSourceXML::ParseXML(ParseType pType, const string &file,
					       const std::unique_ptr<CTPPSPixelDAQMapping> &mapping, const std::unique_ptr<CTPPSPixelAnalysisMask> &mask)
{
  unique_ptr<XercesDOMParser> parser(new XercesDOMParser());
  parser->parse(file.c_str());

  DOMDocument* domDoc = parser->getDocument();

  if (!domDoc)
    throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::ParseXML") << "Cannot parse file `" << file
								      << "' (domDoc = NULL)." << endl;

  DOMElement* elementRoot = domDoc->getDocumentElement();

  if (!elementRoot)
    throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::ParseXML") << "File `" <<
      file << "' is empty." << endl;

  ParseTreePixel(pType, elementRoot, nTop, 0, mapping, mask);


}

//-----------------------------------------------------------------------------------------------------------

void CTPPSPixelDAQMappingESSourceXML::ParseTreePixel(ParseType pType, xercesc::DOMNode * parent, NodeType parentType,
						     unsigned int parentID, const std::unique_ptr<CTPPSPixelDAQMapping>& mapping,
						     const std::unique_ptr<CTPPSPixelAnalysisMask>& mask)
{
#ifdef DEBUG
  printf(">> CTPPSPixelDAQMappingESSourceXML::ParseTreeRP(%s, %u, %u)\n", XMLString::transcode(parent->getNodeName()),
	 parentType, parentID);
#endif

  DOMNodeList *children = parent->getChildNodes();

  for (unsigned int i = 0; i < children->getLength(); i++)
    {
      DOMNode *n = children->item(i);
      if (n->getNodeType() != DOMNode::ELEMENT_NODE)
	continue;

      NodeType type = GetNodeType(n);

#ifdef DEBUG
      printf("\tname = %s, type = %u\n", XMLString::transcode(n->getNodeName()), type);
#endif

    // structure control
      if (!PixelNode(type))
	continue;

      NodeType expectedParentType;
      switch (type)
	{
	case nArm: expectedParentType = nTop; break;
	case nRPStation: expectedParentType = nArm; break;
	case nRPPot: expectedParentType = nRPStation; break;
	case nRPixPlane: expectedParentType = nRPPot; break;
	case nROC: expectedParentType = nRPixPlane; break;
	case nPixel: expectedParentType = nROC; break;
	default: expectedParentType = nUnknown; break;
	}

      if (expectedParentType != parentType)
	{
	  throw cms::Exception("CTPPSPixelDAQMappingESSourceXML") << "Node " << XMLString::transcode(n->getNodeName())
								  << " not allowed within " << XMLString::transcode(parent->getNodeName()) << " block.\n";
	}

    // parse tag attributes
      unsigned int id = 0;
      bool id_set = false;
      bool fullMask = false;
      DOMNamedNodeMap* attr = n->getAttributes();

      for (unsigned int j = 0; j < attr->getLength(); j++)
	{
	  DOMNode *a = attr->item(j);

	  if (!strcmp(XMLString::transcode(a->getNodeName()), "id"))
	    {
	      sscanf(XMLString::transcode(a->getNodeValue()), "%u", &id);
	      id_set = true;
	    }

	  if (!strcmp(XMLString::transcode(a->getNodeName()), "full_mask"))
	    fullMask = (strcmp(XMLString::transcode(a->getNodeValue()), "no") != 0);
	}

    // content control
      if (!id_set)
	throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::ParseTreePixel") << "id not given for element `"
										<< XMLString::transcode(n->getNodeName()) << "'" << endl;

      if (type == nRPixPlane && id > 5)
	throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::ParseTreePixel") <<
	  "Plane IDs range from 0 to 5. id = " << id << " is invalid." << endl;

#ifdef DEBUG
      printf("\tID found: %u\n", id);
#endif

    // store mapping data
      if (pType == pMapping && type == nROC)
	{
	  const CTPPSPixelFramePosition &framepos = ChipFramePosition(n);
	  CTPPSPixelROCInfo rocInfo;
	  rocInfo.roc = id;

	  const unsigned int armIdx = (parentID / 1000) % 10;
	  const unsigned int stIdx = (parentID / 100) % 10;
	  const unsigned int rpIdx = (parentID / 10) % 10;
	  const unsigned int plIdx = parentID % 10;

	  rocInfo.iD = CTPPSPixelDetId(armIdx, stIdx, rpIdx, plIdx);

	  mapping->insert(framepos, rocInfo);

	  continue;
	}

    // store mask data
      if (pType == pMask && type == nROC)
	{
	  const unsigned int armIdx = (parentID / 1000) % 10;
	  const unsigned int stIdx = (parentID / 100) % 10;
	  const unsigned int rpIdx = (parentID / 10) % 10;
	  const unsigned int plIdx = parentID % 10;


	  uint32_t symbId = (id << offsetROCinDetId);

	  symbId |= CTPPSPixelDetId(armIdx, stIdx, rpIdx, plIdx);


	  CTPPSPixelROCAnalysisMask am;
	  am.fullMask = fullMask;
	  GetPixels(n, am.maskedPixels);

	  mask->insert(symbId, am);

	  continue;
	}

    // recursion (deeper in the tree)
      ParseTreePixel(pType, n, type,  parentID * 10 + id, mapping, mask);
    }
}


//----------------------------------------------------------------------------------------------------

CTPPSPixelFramePosition CTPPSPixelDAQMappingESSourceXML::ChipFramePosition(xercesc::DOMNode *chipnode)
{
  CTPPSPixelFramePosition fp;
  unsigned char attributeFlag = 0;

  DOMNamedNodeMap* attr = chipnode->getAttributes();
  for (unsigned int j = 0; j < attr->getLength(); j++)
    {
      DOMNode *a = attr->item(j);

      if (fp.setXMLAttribute(XMLString::transcode(a->getNodeName()), XMLString::transcode(a->getNodeValue()), attributeFlag) > 1)
	{
	  throw cms::Exception("CTPPSPixelDAQMappingESSourceXML") <<
	    "Unrecognized tag `" << XMLString::transcode(a->getNodeName()) <<
	    "' or incompatible value `" << XMLString::transcode(a->getNodeValue()) <<
	    "'." << endl;
	}
    }

  if (!fp.checkXMLAttributeFlag(attributeFlag))
    {
      throw cms::Exception("CTPPSPixelDAQMappingESSourceXML") <<
	"Wrong/incomplete DAQ channel specification (attributeFlag = " << attributeFlag << ")." << endl;
    }

  return fp;
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelDAQMappingESSourceXML::NodeType CTPPSPixelDAQMappingESSourceXML::GetNodeType(xercesc::DOMNode *n)
{
// common node types
  if (Test(n, tagArm)) return nArm;
  if (Test(n, tagROC)) return nROC;

// RP node types
  if (Test(n, tagRPStation)) return nRPStation;
  if (Test(n, tagRPPot)) return nRPPot;
  if (Test(n, tagRPixPlane)) return nRPixPlane;

  throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::GetNodeType") << "Unknown tag `"
								       << XMLString::transcode(n->getNodeName()) << "'.\n";
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelDAQMappingESSourceXML::GetPixels(xercesc::DOMNode *n, set<std::pair<unsigned char, unsigned char> > &pixels)
{
  DOMNodeList *children = n->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++)
    {
      DOMNode *n = children->item(i);
      if (n->getNodeType() != DOMNode::ELEMENT_NODE || !Test(n, "pixel"))
	continue;

      DOMNamedNodeMap* attr = n->getAttributes();
      bool pixelSet = false;
      bool rowSet = false;
      bool colSet = false;
      std::pair<unsigned int, unsigned int> currentPixel;
      for (unsigned int j = 0; j < attr->getLength(); j++)
	{
	  DOMNode *a = attr->item(j);

	  if (!strcmp(XMLString::transcode(a->getNodeName()), "row"))
	    {
	      unsigned int row = 0;
	      sscanf(XMLString::transcode(a->getNodeValue()), "%u", &row);
	      currentPixel.first = row;
	      rowSet = true;
	    }
	  if (!strcmp(XMLString::transcode(a->getNodeName()), "col"))
	    {
	      unsigned int col = 0;
	      sscanf(XMLString::transcode(a->getNodeValue()), "%u", &col);
	      currentPixel.second = col;
	      colSet = true;
	    }

	  pixelSet = rowSet & colSet;
	  if(pixelSet){
	    pixels.insert(currentPixel);
	    break;
	  }
	}



      if (!pixelSet)
	{
	  throw cms::Exception("CTPPSPixelDAQMappingESSourceXML::GetChannels") <<
	    "Channel tags must have a row or col attribute.";
	}
    }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSPixelDAQMappingESSourceXML);
