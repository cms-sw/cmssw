/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*   Marcin Borratynski (mborratynski@gmail.com)
*   Seyed Mohsen Etesami (setesami@cern.ch)
*   Laurent Forthomme
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

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/TotemFramePosition.h"
#include "Utilities/Xerces/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

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
 * \brief Loads TotemDAQMapping and TotemAnalysisMask from two XML files.
 **/
class TotemDAQMappingESSourceXML : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  static const std::string tagVFAT;
  static const std::string tagChannel;
  static const std::string tagAnalysisMask;

  /// Common position tags
  static const std::string tagArm;

  /// RP XML tags
  static const std::string tagRPStation;
  static const std::string tagRPPot;
  static const std::string tagRPPlane;

  /// COMMON Chip XML tags
  static const std::string tagChip1;
  static const std::string tagChip2;

  /// diamond specific tags
  static const std::string tagDiamondPlane;
  static const std::string tagDiamondCh;

  /// totem timing specific tags
  static const std::string tagSampicBoard;
  static const std::string tagSampicCh;
  static const std::string tagTotemTimingCh;
  static const std::string tagTotemTimingPlane;

  /// TOTEM nT2 specific tags
  static const std::string tagTotemT2Plane;
  static const std::string tagTotemT2Tile;

  TotemDAQMappingESSourceXML(const edm::ParameterSet &);
  ~TotemDAQMappingESSourceXML() override;

  edm::ESProducts<std::unique_ptr<TotemDAQMapping>, std::unique_ptr<TotemAnalysisMask>> produce(const TotemReadoutRcd &);

private:
  unsigned int verbosity;

  /// label of the CTPPS sub-system
  string subSystemName;

  //subdetector id for sampic
  unsigned int sampicSubDetId;

  /// the mapping files
  std::vector<std::string> mappingFileNames;

  struct ConfigBlock {
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
  enum NodeType {
    nUnknown,
    nSkip,
    nTop,
    nArm,
    nRPStation,
    nRPPot,
    nRPPlane,
    nDiamondPlane,
    nChip,
    nDiamondCh,
    nChannel,
    nSampicBoard,
    nSampicChannel,
    nTotemTimingPlane,
    nTotemTimingCh,
    nTotemT2Plane,
    nTotemT2Tile
  };

  /// whether to parse a mapping of a mask XML
  enum ParseType { pMapping, pMask };

  /// parses XML file
  void ParseXML(ParseType,
                const string &file,
                const std::unique_ptr<TotemDAQMapping> &,
                const std::unique_ptr<TotemAnalysisMask> &);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeRP(ParseType,
                   xercesc::DOMNode *,
                   NodeType,
                   unsigned int parentID,
                   const std::unique_ptr<TotemDAQMapping> &,
                   const std::unique_ptr<TotemAnalysisMask> &);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeDiamond(ParseType,
                        xercesc::DOMNode *,
                        NodeType,
                        unsigned int parentID,
                        const std::unique_ptr<TotemDAQMapping> &,
                        const std::unique_ptr<TotemAnalysisMask> &);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeTotemTiming(ParseType,
                            xercesc::DOMNode *,
                            NodeType,
                            unsigned int parentID,
                            const std::unique_ptr<TotemDAQMapping> &,
                            const std::unique_ptr<TotemAnalysisMask> &);

  /// recursive method to extract nT2-related information from the DOM tree
  void ParseTreeTotemT2(ParseType,
                        xercesc::DOMNode *,
                        NodeType,
                        unsigned int parentID,
                        const std::unique_ptr<TotemDAQMapping> &,
                        const std::unique_ptr<TotemAnalysisMask> &);

private:
  /// adds the path prefix, if needed
  string CompleteFileName(const string &fn);

  /// returns true iff the node is of the given name
  bool Test(xercesc::DOMNode *node, const std::string &name) {
    return !(name.compare(cms::xerces::toString(node->getNodeName())));
  }

  /// determines node type
  NodeType GetNodeType(xercesc::DOMNode *);

  /// returns the content of the node
  string GetNodeContent(xercesc::DOMNode *parent) { return string(cms::xerces::toString(parent->getTextContent())); }

  /// returns the value of the node
  string GetNodeValue(xercesc::DOMNode *node) { return cms::xerces::toString(node->getNodeValue()); }

  /// extracts VFAT's DAQ channel from XML attributes
  TotemFramePosition ChipFramePosition(xercesc::DOMNode *chipnode);

  void GetChannels(xercesc::DOMNode *n, std::set<unsigned char> &channels);

  bool RPNode(NodeType type) {
    return ((type == nArm) || (type == nRPStation) || (type == nRPPot) || (type == nRPPlane) || (type == nChip));
  }

  bool DiamondNode(NodeType type) {
    return ((type == nArm) || (type == nRPStation) || (type == nRPPot) || (type == nDiamondPlane) ||
            (type == nDiamondCh));
  }

  bool TotemTimingNode(NodeType type) {
    return ((type == nArm) || (type == nRPStation) || (type == nRPPot) || (type == nSampicBoard) ||
            (type == nSampicChannel) || (type == nTotemTimingPlane) || (type == nTotemTimingCh));
  }

  bool TotemT2Node(NodeType type) { return type == nArm || type == nTotemT2Plane || type == nTotemT2Tile; }

  bool CommonNode(NodeType type) { return ((type == nChip) || (type == nArm)); }

protected:
  /// sets infinite validity of this data
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace xercesc;

const string TotemDAQMappingESSourceXML::tagVFAT = "vfat";
const string TotemDAQMappingESSourceXML::tagChannel = "channel";
const string TotemDAQMappingESSourceXML::tagAnalysisMask = "analysisMask";

// common XML position tags
const string TotemDAQMappingESSourceXML::tagArm = "arm";

// common XML Chip tags
const string TotemDAQMappingESSourceXML::tagChip1 = "vfat";
const string TotemDAQMappingESSourceXML::tagChip2 = "test_vfat";

// specific RP XML tags
const string TotemDAQMappingESSourceXML::tagRPStation = "station";
const string TotemDAQMappingESSourceXML::tagRPPot = "rp_detector_set";
const string TotemDAQMappingESSourceXML::tagRPPlane = "rp_plane";

// specific tags for diamond
const string TotemDAQMappingESSourceXML::tagDiamondPlane = "rp_plane_diamond";
const string TotemDAQMappingESSourceXML::tagDiamondCh = "diamond_channel";

// specific tags for totem timing
const string TotemDAQMappingESSourceXML::tagSampicBoard = "rp_sampic_board";
const string TotemDAQMappingESSourceXML::tagSampicCh = "rp_sampic_channel";
const string TotemDAQMappingESSourceXML::tagTotemTimingCh = "timing_channel";
const string TotemDAQMappingESSourceXML::tagTotemTimingPlane = "timing_plane";

// specific tags for TOTEM nT2
const string TotemDAQMappingESSourceXML::tagTotemT2Plane = "nt2_plane";
const string TotemDAQMappingESSourceXML::tagTotemT2Tile = "nt2_tile";

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::TotemDAQMappingESSourceXML(const edm::ParameterSet &conf)
    : verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
      subSystemName(conf.getUntrackedParameter<string>("subSystem")),
      sampicSubDetId(conf.getParameter<unsigned int>("sampicSubDetId")),
      currentBlock(0),
      currentBlockValid(false) {
  for (const auto &it : conf.getParameter<vector<ParameterSet>>("configuration")) {
    ConfigBlock b;
    b.validityRange = it.getParameter<EventRange>("validityRange");
    b.mappingFileNames = it.getParameter<vector<string>>("mappingFileNames");
    b.maskFileNames = it.getParameter<vector<string>>("maskFileNames");
    configuration.push_back(b);
  }

  setWhatProduced(this, subSystemName);
  findingRecord<TotemReadoutRcd>();
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                const edm::IOVSyncValue &iosv,
                                                edm::ValidityInterval &oValidity) {
  LogVerbatim("TotemDAQMappingESSourceXML") << ">> TotemDAQMappingESSourceXML::setIntervalFor(" << key.name() << ")";

  LogVerbatim("TotemDAQMappingESSourceXML")
      << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  currentBlockValid = false;
  for (unsigned int idx = 0; idx < configuration.size(); ++idx) {
    const auto &bl = configuration[idx];

    edm::EventRange range = bl.validityRange;

    // If "<run>:min" is specified in python config, it is translated into event <run>:0:1.
    // However, the truly minimal event id often found in data is <run>:0:0. Therefore the
    // adjustment below is needed.
    if (range.startEventID().luminosityBlock() == 0 && range.startEventID().event() == 1)
      range = edm::EventRange(edm::EventID(range.startEventID().run(), 0, 0), range.endEventID());

    if (edm::contains(range, iosv.eventID())) {
      currentBlockValid = true;
      currentBlock = idx;

      const IOVSyncValue begin(range.startEventID());
      const IOVSyncValue end(range.endEventID());
      oValidity = edm::ValidityInterval(begin, end);

      LogVerbatim("TotemDAQMappingESSourceXML") << "    block found: index=" << currentBlock << ", interval=("
                                                << range.startEventID() << " - " << range.endEventID() << ")";

      return;
    }
  }

  if (!currentBlockValid) {
    throw cms::Exception("TotemDAQMappingESSourceXML::setIntervalFor")
        << "No configuration for event " << iosv.eventID();
  }
}

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::~TotemDAQMappingESSourceXML() {}

//----------------------------------------------------------------------------------------------------

string TotemDAQMappingESSourceXML::CompleteFileName(const string &fn) {
  FileInPath fip(fn);
  return fip.fullPath();
}

//----------------------------------------------------------------------------------------------------
static inline std::string to_string(const XMLCh *ch) { return XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(ch); }

edm::ESProducts<std::unique_ptr<TotemDAQMapping>, std::unique_ptr<TotemAnalysisMask>>
TotemDAQMappingESSourceXML::produce(const TotemReadoutRcd &) {
  assert(currentBlockValid);

  auto mapping = std::make_unique<TotemDAQMapping>();
  auto mask = std::make_unique<TotemAnalysisMask>();

  try {
    // initialize Xerces
    cms::concurrency::xercesInitialize();

    // load mapping files
    for (const auto &fn : configuration[currentBlock].mappingFileNames)
      ParseXML(pMapping, CompleteFileName(fn), mapping, mask);

    // load mask files
    for (const auto &fn : configuration[currentBlock].maskFileNames)
      ParseXML(pMask, CompleteFileName(fn), mapping, mask);

    // release Xerces
    cms::concurrency::xercesTerminate();
  } catch (const XMLException &e) {
    throw cms::Exception("XMLDocument") << "cms::concurrency::xercesInitialize failed because of "
                                        << to_string(e.getMessage()) << std::endl;
  } catch (const SAXException &e) {
    throw cms::Exception("XMLDocument") << "XML parser reported: " << to_string(e.getMessage()) << "." << std::endl;
  }

  // commit the products
  return edm::es::products(std::move(mapping), std::move(mask));
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseXML(ParseType pType,
                                          const string &file,
                                          const std::unique_ptr<TotemDAQMapping> &mapping,
                                          const std::unique_ptr<TotemAnalysisMask> &mask) {
  unique_ptr<XercesDOMParser> parser(new XercesDOMParser());
  parser->parse(file.c_str());

  DOMDocument *domDoc = parser->getDocument();

  if (!domDoc)
    throw cms::Exception("TotemDAQMappingESSourceXML::ParseXML")
        << "Cannot parse file `" << file << "' (domDoc = NULL).";

  DOMElement *elementRoot = domDoc->getDocumentElement();

  if (!elementRoot)
    throw cms::Exception("TotemDAQMappingESSourceXML::ParseXML") << "File `" << file << "' is empty.";

  ParseTreeRP(pType, elementRoot, nTop, 0, mapping, mask);

  ParseTreeDiamond(pType, elementRoot, nTop, 0, mapping, mask);

  ParseTreeTotemTiming(pType, elementRoot, nTop, 0, mapping, mask);

  ParseTreeTotemT2(pType, elementRoot, nTop, 0, mapping, mask);
}

//-----------------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeRP(ParseType pType,
                                             xercesc::DOMNode *parent,
                                             NodeType parentType,
                                             unsigned int parentID,
                                             const std::unique_ptr<TotemDAQMapping> &mapping,
                                             const std::unique_ptr<TotemAnalysisMask> &mask) {
#ifdef DEBUG
  printf(">> TotemDAQMappingESSourceXML::ParseTreeRP(%s, %u, %u)\n",
         cms::xerces::toString(parent->getNodeName()),
         parentType,
         parentID);
#endif

  DOMNodeList *children = parent->getChildNodes();

  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    NodeType type = GetNodeType(n);

#ifdef DEBUG
    printf("\tname = %s, type = %u\n", cms::xerces::toString(n->getNodeName()), type);
#endif

    // structure control
    if (!RPNode(type))
      continue;

    NodeType expectedParentType;
    switch (type) {
      case nArm:
        expectedParentType = nTop;
        break;
      case nRPStation:
        expectedParentType = nArm;
        break;
      case nRPPot:
        expectedParentType = nRPStation;
        break;
      case nRPPlane:
        expectedParentType = nRPPot;
        break;
      case nChip:
        expectedParentType = nRPPlane;
        break;
      case nChannel:
        expectedParentType = nChip;
        break;
      default:
        expectedParentType = nUnknown;
        break;
    }

    if (expectedParentType != parentType) {
      throw cms::Exception("TotemDAQMappingESSourceXML")
          << "Node " << cms::xerces::toString(n->getNodeName()) << " not allowed within "
          << cms::xerces::toString(parent->getNodeName()) << " block.\n";
    }

    // parse tag attributes
    unsigned int id = 0, hw_id = 0;
    bool id_set = false, hw_id_set = false;
    bool fullMask = false;
    DOMNamedNodeMap *attr = n->getAttributes();

    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &id);
        id_set = true;
      }

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "hw_id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%x", &hw_id);
        hw_id_set = true;
      }

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "full_mask"))
        fullMask = (strcmp(cms::xerces::toString(a->getNodeValue()).c_str(), "no") != 0);
    }

    // content control
    if (!id_set)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP")
          << "id not given for element `" << cms::xerces::toString(n->getNodeName()) << "'";

    if (!hw_id_set && type == nChip && pType == pMapping)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP")
          << "hw_id not given for element `" << cms::xerces::toString(n->getNodeName()) << "'";

    if (type == nRPPlane && id > 9)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP")
          << "Plane IDs range from 0 to 9. id = " << id << " is invalid.";

#ifdef DEBUG
    printf("\tID found: 0x%x\n", id);
#endif

    // store mapping data
    if (pType == pMapping && type == nChip) {
      const TotemFramePosition &framepos = ChipFramePosition(n);
      TotemVFATInfo vfatInfo;
      vfatInfo.hwID = hw_id;

      const unsigned int armIdx = (parentID / 1000) % 10;
      const unsigned int stIdx = (parentID / 100) % 10;
      const unsigned int rpIdx = (parentID / 10) % 10;
      const unsigned int plIdx = parentID % 10;

      vfatInfo.symbolicID.symbolicID = TotemRPDetId(armIdx, stIdx, rpIdx, plIdx, id);

      mapping->insert(framepos, vfatInfo);

      continue;
    }

    // store mask data
    if (pType == pMask && type == nChip) {
      const unsigned int armIdx = (parentID / 1000) % 10;
      const unsigned int stIdx = (parentID / 100) % 10;
      const unsigned int rpIdx = (parentID / 10) % 10;
      const unsigned int plIdx = parentID % 10;

      TotemSymbID symbId;
      symbId.symbolicID = TotemRPDetId(armIdx, stIdx, rpIdx, plIdx, id);

      TotemVFATAnalysisMask am;
      am.fullMask = fullMask;
      GetChannels(n, am.maskedChannels);

      mask->insert(symbId, am);

      continue;
    }

    // recursion (deeper in the tree)
    ParseTreeRP(pType, n, type, parentID * 10 + id, mapping, mask);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeDiamond(ParseType pType,
                                                  xercesc::DOMNode *parent,
                                                  NodeType parentType,
                                                  unsigned int parentID,
                                                  const std::unique_ptr<TotemDAQMapping> &mapping,
                                                  const std::unique_ptr<TotemAnalysisMask> &mask) {
#ifdef DEBUG
  printf(">> TotemDAQMappingESSourceXML::ParseTreeDiamond(%s, %u, %u)\n",
         cms::xerces::toString(parent->getNodeName()),
         parentType,
         parentID);
#endif

  DOMNodeList *children = parent->getChildNodes();

  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    NodeType type = GetNodeType(n);
#ifdef DEBUG
    printf("\tname = %s, type = %u\n", cms::xerces::toString(n->getNodeName()), type);
#endif

    // structure control
    if (!DiamondNode(type))
      continue;

    NodeType expectedParentType;
    switch (type) {
      case nArm:
        expectedParentType = nTop;
        break;
      case nRPStation:
        expectedParentType = nArm;
        break;
      case nRPPot:
        expectedParentType = nRPStation;
        break;
      case nDiamondPlane:
        expectedParentType = nRPPot;
        break;
      case nDiamondCh:
        expectedParentType = nDiamondPlane;
        break;
      default:
        expectedParentType = nUnknown;
        break;
    }

    if (expectedParentType != parentType) {
      throw cms::Exception("TotemDAQMappingESSourceXML")
          << "Node " << cms::xerces::toString(n->getNodeName()) << " not allowed within "
          << cms::xerces::toString(parent->getNodeName()) << " block.\n";
    }

    // parse tag attributes
    unsigned int id = 0, hw_id = 0;
    bool id_set = false, hw_id_set = false;
    DOMNamedNodeMap *attr = n->getAttributes();

    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &id);
        id_set = true;
      }

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "hw_id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%x", &hw_id);
        hw_id_set = true;
      }
    }

    // content control
    if (!id_set)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeDiamond")
          << "id not given for element `" << cms::xerces::toString(n->getNodeName()) << "'";

    if (!hw_id_set && type == nDiamondCh && pType == pMapping)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeDiamond")
          << "hw_id not given for element `" << cms::xerces::toString(n->getNodeName()) << "'";

    if (type == nDiamondPlane && id > 3)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeDiamond")
          << "Plane IDs range from 0 to 3. id = " << id << " is invalid.";

#ifdef DEBUG
    printf("\tID found: 0x%x\n", id);
#endif

    // store mapping data
    if (pType == pMapping && type == nDiamondCh) {
      const TotemFramePosition &framepos = ChipFramePosition(n);

      TotemVFATInfo vfatInfo;
      vfatInfo.hwID = hw_id;

      if (type == nDiamondCh) {
        unsigned int ArmNum = (parentID / 10000) % 10;
        unsigned int StationNum = (parentID / 1000) % 10;
        unsigned int RpNum = (parentID / 100) % 10;
        unsigned int PlaneNum = (parentID % 100);

        vfatInfo.symbolicID.symbolicID = CTPPSDiamondDetId(ArmNum, StationNum, RpNum, PlaneNum, id);
      }

      mapping->insert(framepos, vfatInfo);

      continue;
    }

    unsigned int childId;
    if (pType == pMapping && type == nDiamondPlane)
      childId = parentID * 100 + id;
    else
      childId = parentID * 10 + id;

    ParseTreeDiamond(pType, n, type, childId, mapping, mask);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeTotemTiming(ParseType pType,
                                                      xercesc::DOMNode *parent,
                                                      NodeType parentType,
                                                      unsigned int parentID,
                                                      const std::unique_ptr<TotemDAQMapping> &mapping,
                                                      const std::unique_ptr<TotemAnalysisMask> &mask) {
  DOMNodeList *children = parent->getChildNodes();

  // Fill map hwId -> TotemTimingPlaneChannelPair
  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *child = children->item(i);
    if ((child->getNodeType() != DOMNode::ELEMENT_NODE) || (GetNodeType(child) != nTotemTimingCh))
      continue;

    int plane = -1;
    DOMNamedNodeMap *attr = parent->getAttributes();
    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id"))
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%d", &plane);
    }

    int channel = -1;
    unsigned int hwId = 0;
    attr = child->getAttributes();
    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id"))
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%d", &channel);
      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "hwId"))
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%x", &hwId);
    }

    mapping->totemTimingChannelMap[(uint8_t)hwId] = TotemDAQMapping::TotemTimingPlaneChannelPair(plane, channel);
  }

  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    NodeType type = GetNodeType(n);

    // structure control
    if (!TotemTimingNode(type))
      continue;

    NodeType expectedParentType;
    switch (type) {
      case nArm:
        expectedParentType = nTop;
        break;
      case nRPStation:
        expectedParentType = nArm;
        break;
      case nRPPot:
        expectedParentType = nRPStation;
        break;
      case nSampicBoard:
        expectedParentType = nRPPot;
        break;
      case nSampicChannel:
        expectedParentType = nSampicBoard;
        break;
      case nTotemTimingPlane:
        expectedParentType = nRPPot;
        break;
      case nTotemTimingCh:
        expectedParentType = nTotemTimingPlane;
        break;
      default:
        expectedParentType = nUnknown;
        break;
    }

    if (expectedParentType != parentType) {
      throw cms::Exception("TotemDAQMappingESSourceXML")
          << "Node " << cms::xerces::toString(n->getNodeName()) << " not allowed within "
          << cms::xerces::toString(parent->getNodeName()) << " block.\n";
    }

    // parse tag attributes
    unsigned int id = 0;
    bool id_set = false;
    DOMNamedNodeMap *attr = n->getAttributes();

    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &id);
        id_set = true;
      }
    }

    // content control
    if (!id_set)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeTotemTiming")
          << "id not given for element `" << cms::xerces::toString(n->getNodeName()) << "'";
    if (type == nSampicBoard && id > 5)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeTotemTiming")
          << "SampicBoard IDs range from 0 to 5. id = " << id << " is invalid.";

    // store mapping data
    if (pType == pMapping && type == nSampicChannel) {
      const TotemFramePosition &framepos = ChipFramePosition(n);

      TotemVFATInfo vfatInfo;
      unsigned int ArmNum = (parentID / 10000) % 10;
      unsigned int StationNum = (parentID / 1000) % 10;
      unsigned int RpNum = (parentID / 100) % 10;

      vfatInfo.symbolicID.symbolicID = TotemTimingDetId(ArmNum,
                                                        StationNum,
                                                        RpNum,
                                                        0,
                                                        TotemTimingDetId::ID_NOT_SET,
                                                        sampicSubDetId);  //Dynamical: it is encoded in the frame

      mapping->insert(framepos, vfatInfo);

      continue;
    }

    unsigned int childId;
    if (pType == pMapping && type == nSampicBoard)
      childId = parentID * 100 + id;
    else
      childId = parentID * 10 + id;

    ParseTreeTotemTiming(pType, n, type, childId, mapping, mask);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeTotemT2(ParseType pType,
                                                  xercesc::DOMNode *parent,
                                                  NodeType parentType,
                                                  unsigned int parentID,
                                                  const std::unique_ptr<TotemDAQMapping> &mapping,
                                                  const std::unique_ptr<TotemAnalysisMask> &mask) {
  DOMNodeList *children = parent->getChildNodes();

  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *child = children->item(i);
    if (child->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    NodeType type = GetNodeType(child);

    // structure control
    if (!TotemT2Node(type))
      continue;

    NodeType expectedParentType;
    switch (type) {
      case nArm:
        expectedParentType = nTop;
        break;
      case nTotemT2Plane:
        expectedParentType = nArm;
        break;
      case nTotemT2Tile:
        expectedParentType = nTotemT2Plane;
        break;
      default:
        expectedParentType = nUnknown;
        break;
    }

    if (expectedParentType != parentType) {
      throw cms::Exception("TotemDAQMappingESSourceXML")
          << "Node " << cms::xerces::toString(child->getNodeName()) << " not allowed within "
          << cms::xerces::toString(parent->getNodeName()) << " block.\n";
    }

    unsigned int id = 0;
    bool id_set = false;
    DOMNamedNodeMap *attr = child->getAttributes();

    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);
      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id")) {
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &id);
        id_set = true;
      }
    }
    if (pType == pMapping && type == nTotemT2Tile) {
      // parse tag attributes
      unsigned int hw_id = 0;
      bool hw_id_set = false;
      DOMNamedNodeMap *attr = child->getAttributes();

      for (unsigned int j = 0; j < attr->getLength(); j++) {
        DOMNode *a = attr->item(j);
        if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "hwId")) {
          sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &hw_id);
          hw_id_set = true;
        }
      }

      // content control
      if (!id_set)
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeTotemT2")
            << "id not given for element `" << cms::xerces::toString(child->getNodeName()) << "'";
      if (!hw_id_set)
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeTotemT2")
            << "hwId not given for element `" << cms::xerces::toString(child->getNodeName()) << "'";

      // store mapping data
      const TotemFramePosition &framepos = ChipFramePosition(child);
      TotemVFATInfo vfatInfo;
      vfatInfo.hwID = hw_id;
      unsigned int arm = parentID / 10, plane = parentID % 10;
      vfatInfo.symbolicID.symbolicID = TotemT2DetId(arm, plane, id);

      mapping->insert(framepos, vfatInfo);

      continue;
    }
    // follow tree recursively
    ParseTreeTotemT2(pType, child, type, parentID * 10 + id, mapping, mask);
  }
}

//----------------------------------------------------------------------------------------------------

TotemFramePosition TotemDAQMappingESSourceXML::ChipFramePosition(xercesc::DOMNode *chipnode) {
  TotemFramePosition fp;
  unsigned char attributeFlag = 0;

  DOMNamedNodeMap *attr = chipnode->getAttributes();
  for (unsigned int j = 0; j < attr->getLength(); j++) {
    DOMNode *a = attr->item(j);
    if (fp.setXMLAttribute(
            cms::xerces::toString(a->getNodeName()), cms::xerces::toString(a->getNodeValue()), attributeFlag) > 1) {
      throw cms::Exception("TotemDAQMappingESSourceXML")
          << "Unrecognized tag `" << cms::xerces::toString(a->getNodeName()) << "' or incompatible value `"
          << cms::xerces::toString(a->getNodeValue()) << "'.";
    }
  }

  if (!fp.checkXMLAttributeFlag(attributeFlag)) {
    throw cms::Exception("TotemDAQMappingESSourceXML")
        << "Wrong/incomplete DAQ channel specification (attributeFlag = " << attributeFlag << ").";
  }

  return fp;
}

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::NodeType TotemDAQMappingESSourceXML::GetNodeType(xercesc::DOMNode *n) {
  // common node types
  if (Test(n, tagArm))
    return nArm;
  if (Test(n, tagChip1))
    return nChip;
  if (Test(n, tagChip2))
    return nChip;

  // RP node types
  if (Test(n, tagRPStation))
    return nRPStation;
  if (Test(n, tagRPPot))
    return nRPPot;
  if (Test(n, tagRPPlane))
    return nRPPlane;

  //diamond specifics
  if (Test(n, tagDiamondCh))
    return nDiamondCh;
  if (Test(n, tagDiamondPlane))
    return nDiamondPlane;

  //totem timing specifics
  if (Test(n, tagSampicBoard))
    return nSampicBoard;
  if (Test(n, tagSampicCh))
    return nSampicChannel;
  if (Test(n, tagTotemTimingCh))
    return nTotemTimingCh;
  if (Test(n, tagTotemTimingPlane))
    return nTotemTimingPlane;

  // TOTEM nT2 specifics
  if (Test(n, tagTotemT2Plane))
    return nTotemT2Plane;
  if (Test(n, tagTotemT2Tile))
    return nTotemT2Tile;

  // for backward compatibility
  if (Test(n, "trigger_vfat"))
    return nSkip;

  throw cms::Exception("TotemDAQMappingESSourceXML::GetNodeType")
      << "Unknown tag `" << cms::xerces::toString(n->getNodeName()) << "'.\n";
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::GetChannels(xercesc::DOMNode *n, set<unsigned char> &channels) {
  DOMNodeList *children = n->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE || !Test(n, "channel"))
      continue;

    DOMNamedNodeMap *attr = n->getAttributes();
    bool idSet = false;
    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);

      if (!strcmp(cms::xerces::toString(a->getNodeName()).c_str(), "id")) {
        unsigned int id = 0;
        sscanf(cms::xerces::toString(a->getNodeValue()).c_str(), "%u", &id);
        channels.insert(id);
        idSet = true;
        break;
      }
    }

    if (!idSet) {
      throw cms::Exception("TotemDAQMappingESSourceXML::GetChannels") << "Channel tags must have an `id' attribute.";
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(TotemDAQMappingESSourceXML);
