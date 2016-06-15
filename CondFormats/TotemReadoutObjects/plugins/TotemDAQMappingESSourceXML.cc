/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*   Marcin Borratynski (mborratynski@gmail.com)
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

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemFramePosition.h"

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
class TotemDAQMappingESSourceXML: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
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

  /// T2 XML tags
  static const std::string tagT2;
  static const std::string tagT2Half;
  static const std::string tagT2detector;

  /// T1 XML tags
  static const std::string tagT1;
  static const std::string tagT1Arm;
  static const std::string tagT1Plane;
  static const std::string tagT1CSC;
  static const std::string tagT1ChannelType;

  /// COMMON Chip XML tags
  static const std::string tagChip1;
  static const std::string tagChip2;
  static const std::string tagTriggerVFAT1;

  TotemDAQMappingESSourceXML(const edm::ParameterSet &);
  ~TotemDAQMappingESSourceXML();

  edm::ESProducts< boost::shared_ptr<TotemDAQMapping>, boost::shared_ptr<TotemAnalysisMask> > produce( const TotemReadoutRcd & );

private:
  unsigned int verbosity;

  /// the mapping files
  std::vector<std::string> mappingFileNames;

  /// the mask files
  std::vector<std::string> maskFileNames;

  /// enumeration of XML node types
  enum NodeType { nUnknown, nTop, nArm, nRPStation, nRPPot, nRPPlane, nChip, nTriggerVFAT,
    nT2, nT2Half, nT2Det, nT1, nT1Arm, nT1Plane, nT1CSC, nT1ChannelType, nChannel };

  /// whether to parse a mapping of a mask XML
  enum ParseType { pMapping, pMask };

  /// parses XML file
  void ParseXML(ParseType, const string &file, const boost::shared_ptr<TotemDAQMapping>&, const boost::shared_ptr<TotemAnalysisMask>&);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeRP(ParseType, xercesc::DOMNode *, NodeType, unsigned int parentID,
    const boost::shared_ptr<TotemDAQMapping>&, const boost::shared_ptr<TotemAnalysisMask>&);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeT1(ParseType, xercesc::DOMNode *, NodeType, unsigned int parentID,
    const boost::shared_ptr<TotemDAQMapping>&, const boost::shared_ptr<TotemAnalysisMask>&,
    unsigned int T1Arm, unsigned int T1Plane, unsigned int T1CSC);

  /// recursive method to extract RP-related information from the DOM tree
  void ParseTreeT2(ParseType, xercesc::DOMNode *, NodeType, unsigned int parentID,
    const boost::shared_ptr<TotemDAQMapping>&, const boost::shared_ptr<TotemAnalysisMask>&);

private:
  /// adds the path prefix, if needed
  string CompleteFileName(const string &fn);

  /// returns the top element from an XML file
  xercesc::DOMDocument* GetDOMDocument(string file);

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
  TotemFramePosition ChipFramePosition(xercesc::DOMNode *chipnode);

  void GetChannels(xercesc::DOMNode *n, std::set<unsigned char> &channels);

  bool RPNode(NodeType type)
  {
    return ((type == nArm)||(type == nRPStation)||(type == nRPPot)||(type == nRPPlane)||(type == nChip)||(type == nTriggerVFAT));
  }

  bool T2Node(NodeType type)
  {
    return ((type==nT2)||(type==nT2Det)|| (type==nT2Half));
  }

  bool T1Node(NodeType type)
  {
    return ((type==nT1)||(type==nT1Arm)|| (type==nT1Plane) || (type==nT1CSC) || (type==nT1ChannelType));
  }

  bool CommonNode(NodeType type)
  {
    return ((type==nChip)||(type==nArm));
  }

protected:
  /// sets infinite validity of this data
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&);
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;
using namespace xercesc;

const string TotemDAQMappingESSourceXML::tagVFAT="vfat";
const string TotemDAQMappingESSourceXML::tagChannel="channel";
const string TotemDAQMappingESSourceXML::tagAnalysisMask="analysisMask";

// common XML position tags
const string TotemDAQMappingESSourceXML::tagArm = "arm";

// common XML Chip tags
const string TotemDAQMappingESSourceXML::tagChip1 = "vfat";
const string TotemDAQMappingESSourceXML::tagChip2 = "test_vfat";
const string TotemDAQMappingESSourceXML::tagTriggerVFAT1 = "trigger_vfat";

// specific RP XML tags
const string TotemDAQMappingESSourceXML::tagRPStation = "station";
const string TotemDAQMappingESSourceXML::tagRPPot = "rp_detector_set";
const string TotemDAQMappingESSourceXML::tagRPPlane = "rp_plane";

// specific T2 XML tags
const string TotemDAQMappingESSourceXML::tagT2="t2_detector_set";
const string TotemDAQMappingESSourceXML::tagT2detector="t2_detector";
const string TotemDAQMappingESSourceXML::tagT2Half="t2_half";

// specific T1 XML tags
const string TotemDAQMappingESSourceXML::tagT1="t1_detector_set";
const string TotemDAQMappingESSourceXML::tagT1Arm="t1_arm";
const string TotemDAQMappingESSourceXML::tagT1Plane="t1_plane";
const string TotemDAQMappingESSourceXML::tagT1CSC="t1_csc";
const string TotemDAQMappingESSourceXML::tagT1ChannelType="t1_channel_type";

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::TotemDAQMappingESSourceXML(const edm::ParameterSet& conf) :
  verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
  mappingFileNames(conf.getUntrackedParameter< vector<string> >("mappingFileNames")),
  maskFileNames(conf.getUntrackedParameter< vector<string> >("maskFileNames"))
{
  setWhatProduced(this);
  findingRecord<TotemReadoutRcd>();
}

//----------------------------------------------------------------------------------------------------

string TotemDAQMappingESSourceXML::CompleteFileName(const string &fn)
{
  FileInPath fip(fn);
  return fip.fullPath();
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
  const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

//----------------------------------------------------------------------------------------------------

DOMDocument* TotemDAQMappingESSourceXML::GetDOMDocument(string file)
{
  XercesDOMParser* parser = new XercesDOMParser();
  parser->parse(file.c_str());

  DOMDocument* xmlDoc = parser->getDocument();

  if (!xmlDoc)
    throw cms::Exception("TotemDAQMappingESSourceXML::GetDOMDocument") << "Cannot parse file `" << file
      << "' (xmlDoc = NULL)." << endl;

  return xmlDoc;
}

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::~TotemDAQMappingESSourceXML()
{ 
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts< boost::shared_ptr<TotemDAQMapping>, boost::shared_ptr<TotemAnalysisMask> >
  TotemDAQMappingESSourceXML::produce( const TotemReadoutRcd & )
{
  boost::shared_ptr<TotemDAQMapping> mapping(new TotemDAQMapping());
  boost::shared_ptr<TotemAnalysisMask> mask(new TotemAnalysisMask());

  // initialize Xerces
  try
  {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch)
  {
    char* message = XMLString::transcode(toCatch.getMessage());
    throw cms::Exception("TotemDAQMappingESSourceXML") << "An XMLException caught with message: " << message << ".\n";
    XMLString::release(&message);
  }

  // load mapping files
  for (unsigned int i = 0; i < mappingFileNames.size(); ++i)
    ParseXML(pMapping, CompleteFileName(mappingFileNames[i]), mapping, mask);

  // load mask files
  for (unsigned int i = 0; i < maskFileNames.size(); ++i)
    ParseXML(pMask, CompleteFileName(maskFileNames[i]), mapping, mask);

  // release Xerces
  XMLPlatformUtils::Terminate();

  // commit the products
  return edm::es::products(mapping, mask);
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseXML(ParseType pType, const string &file,
  const boost::shared_ptr<TotemDAQMapping> &mapping, const boost::shared_ptr<TotemAnalysisMask> &mask)
{
  DOMDocument* domDoc = GetDOMDocument(file);
  DOMElement* elementRoot = domDoc->getDocumentElement();

  if (!elementRoot)
    throw cms::Exception("TotemDAQMappingESSourceXML::ParseMappingXML") << "File `" <<
      file << "' is empty." << endl;

  ParseTreeRP(pType, elementRoot, nTop, 0, mapping, mask);
  ParseTreeT2(pType, elementRoot, nTop, 0, mapping, mask);
  ParseTreeT1(pType, elementRoot, nTop, 0, mapping, mask, 0,0,0);
}

//-----------------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeRP(ParseType pType, xercesc::DOMNode * parent, NodeType parentType,
  unsigned int parentID, const boost::shared_ptr<TotemDAQMapping>& mapping,
  const boost::shared_ptr<TotemAnalysisMask>& mask)
{
#ifdef DEBUG
  printf(">> TotemDAQMappingESSourceXML::ParseTreeRP(%s, %u, %u)\n", XMLString::transcode(parent->getNodeName()),
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
    if (!RPNode(type))
      continue;

    if ((type != parentType + 1)&&(parentType != nRPPot || type != nTriggerVFAT))
    {
      if (parentType == nTop && type == nRPPot)
      {
	    LogPrint("TotemDAQMappingESSourceXML") << ">> TotemDAQMappingESSourceXML::ParseTreeRP > Warning: tag `" << tagRPPot
					<< "' found in global scope, assuming station ID = 12.";
	    parentID = 12;
      } else {
        throw cms::Exception("TotemDAQMappingESSourceXML") << "Node " << XMLString::transcode(n->getNodeName())
          << " not allowed within " << XMLString::transcode(parent->getNodeName()) << " block.\n";
      }
    }

    // parse tag attributes
    unsigned int id = 0, hw_id = 0;
    bool id_set = false, hw_id_set = false;
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

      if (!strcmp(XMLString::transcode(a->getNodeName()), "hw_id"))
      {
        sscanf(XMLString::transcode(a->getNodeValue()), "%x", &hw_id);
        hw_id_set = true;
      }

      if (!strcmp(XMLString::transcode(a->getNodeName()), "full_mask"))
        fullMask = (strcmp(XMLString::transcode(a->getNodeValue()), "no") != 0);
    }

    // content control
    if (!id_set && type != nTriggerVFAT)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP") << "id not given for element `"
       << XMLString::transcode(n->getNodeName()) << "'" << endl;

    if (!hw_id_set && type == nChip && pType == pMapping)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP") << "hw_id not given for element `"
       << XMLString::transcode(n->getNodeName()) << "'" << endl;

    if (type == nRPPlane && id > 9)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeRP") <<
        "Plane IDs range from 0 to 9. id = " << id << " is invalid." << endl;

#ifdef DEBUG
    printf("\tID found: 0x%x\n", id);
#endif

    // store mapping data
    if (pType == pMapping && (type == nChip || type == nTriggerVFAT))
    {
      const TotemFramePosition &framepos = ChipFramePosition(n);
      TotemVFATInfo vfatInfo;
      vfatInfo.hwID = hw_id;
      vfatInfo.symbolicID.subSystem = TotemSymbID::RP;

      if (type == nChip)
      {
        vfatInfo.symbolicID.symbolicID = parentID * 10 + id;
        vfatInfo.type = TotemVFATInfo::data;
      }

      if (type == nTriggerVFAT)
      {
        vfatInfo.symbolicID.symbolicID = parentID;
        vfatInfo.type = TotemVFATInfo::CC;
      }

      mapping->insert(framepos, vfatInfo);

      continue;
    }

    // store mask data
    if (pType == pMask && type == nChip)
    {
      TotemSymbID symbId;
      symbId.subSystem = TotemSymbID::RP;
      symbId.symbolicID = parentID * 10 + id;

      TotemVFATAnalysisMask am;
      am.fullMask = fullMask;
      GetChannels(n, am.maskedChannels);

      mask->insert(symbId, am);

      continue;
    }

    // recursion (deeper in the tree)
    ParseTreeRP(pType, n, type,  parentID * 10 + id, mapping, mask);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeT2(ParseType pType, xercesc::DOMNode * parent, NodeType parentType,
  unsigned int parentID, const boost::shared_ptr<TotemDAQMapping>& data,
  const boost::shared_ptr<TotemAnalysisMask>& mask)
{
  DOMNodeList *children = parent->getChildNodes();

#ifdef DEBUG
  printf(">> ParseTreeT2(parent,parentType,parentID)=(%p, %i, %u)\n", parent, parentType, parentID);
  printf("\tchildren: Numero children: %li\n", children->getLength());
#endif

  for (unsigned int i = 0; i < children->getLength(); i++)
  {
    DOMNode *n = children->item(i);

    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    // get node type for RP or T2
    NodeType type = GetNodeType(n);

#ifdef DEBUG
    printf("\t\tchildren #%i: is a %s, (of type %i) \n", i, XMLString::transcode(n->getNodeName()), type);
#endif

    if ((type == nUnknown)) {
#ifdef DEBUG
      printf("Found Unknown tag during T2 reading.. EXIT ");
#endif   
      continue;
    }

    if ((T2Node(type)==false)&&(CommonNode(type)==false)) {
#ifdef DEBUG
      printf("Found Non-T2 tag during T2 reading.. EXIT ");
      printf("\t The tag is:  %s \n", XMLString::transcode(n->getNodeName()));
#endif
      continue;
    }

    // get ID_t2 and position

    // id  for T2 plane goes from 0..9; for chip is the 16 bit ID
    // position_t2 was the S-link for chip and for the plane should be a number compatible with arm,ht,pl,pls or HS position
    int ID_t2 = 0;

    unsigned int position_t2 = 0;

    unsigned int arm=0,ht=0,pl=0,pls=0;

    bool idSet_t2 = false;
    //position_t2Set = false;
    int attribcounter_t2planedescript=0;
    unsigned int toaddForParentID=0;

    unsigned hw_id = 0;
    bool hw_id_set = false;

    DOMNamedNodeMap* attr = n->getAttributes();

    //    Begin loop for save T2 element attriute
    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode *a = attr->item(j);
      if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
        sscanf(XMLString::transcode(a->getNodeValue()), "%i", &ID_t2);
        idSet_t2 = true;
      }

      if (!strcmp(XMLString::transcode(a->getNodeName()), "hw_id")) {
        sscanf(XMLString::transcode(a->getNodeValue()), "%x", &hw_id);
        hw_id_set = true;
      }

      if (!strcmp(XMLString::transcode(a->getNodeName()), "position")) {
        position_t2 = atoi(XMLString::transcode(a->getNodeValue()));
        if (pType == pMask)
          toaddForParentID = position_t2;
//        position_t2Set = true;
      }

      if (type == nArm) {
        // arm is the top node and should be reset to 0.
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          parentID=0;
          unsigned int id_arm = atoi(XMLString::transcode(a->getNodeValue()));
          toaddForParentID=20*id_arm;
        }
      }

      if (type == nT2Half) {
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          unsigned int id_half = atoi(XMLString::transcode(a->getNodeValue()));
          toaddForParentID=10*id_half;
        }
      }

      // This is needed in principle only for the old formats
      if(type == nT2Det) {
        if (!strcmp(XMLString::transcode(a->getNodeName()), "arm")) {
          sscanf(XMLString::transcode(a->getNodeValue()), "%u", &arm);
          attribcounter_t2planedescript++;
        }

        if (!strcmp(XMLString::transcode(a->getNodeName()), "ht")) {
          sscanf(XMLString::transcode(a->getNodeValue()), "%u", &ht);
          attribcounter_t2planedescript++;
        }

        if (!strcmp(XMLString::transcode(a->getNodeName()), "pl")) {
          sscanf(XMLString::transcode(a->getNodeValue()), "%u", &pl);
          attribcounter_t2planedescript++;
        }

        if (!strcmp(XMLString::transcode(a->getNodeName()), "pls")) {
          sscanf(XMLString::transcode(a->getNodeValue()), "%u", &pls);
          attribcounter_t2planedescript++;
        }

        // remember id in monitor goes from 0 -- 39
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          // Id saved another time ... just to increment attribcounter
          sscanf(XMLString::transcode(a->getNodeValue()), "%i", &ID_t2);
          attribcounter_t2planedescript++;
        }

        if (!strcmp(XMLString::transcode(a->getNodeName()), "position")) {
          sscanf(XMLString::transcode(a->getNodeValue()), "%u", &position_t2);
          attribcounter_t2planedescript++;
          // Just another indication for further checking. This attribute was not compulsory in monitor.
          attribcounter_t2planedescript=attribcounter_t2planedescript+20;
          // 20 is just a "big number"
        }
      }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // When a plane tag is found, calculate Parent-Id and allows different xml formats:
    // Provide compatibility with old monitor formats.

    // Note:
    // plane position and id foreseen in  final ip5 mapping.
    // plane position NOT foreseen in Monitor.

    if (pType == pMapping) {
      if (type == nT2Det) {
      // Calculate the parent-id from attributes or xml STRUCTURE
//        position_t2Set = true;
        if (attribcounter_t2planedescript>=(21)) {
          // there is already in xml plane the  "position" attribute + other attributes. It is assumed to utilize the parent info

#ifdef DEBUG  
          printf("TotemDAQMappingESSourceXML: attribcounter_t2planedescript: %i \n",attribcounter_t2planedescript);
#endif

          if (attribcounter_t2planedescript>=25) {
            edm::LogVerbatim("TotemDAQMappingESSourceXML") << "T2-Plane attribute utilezed for parentID: position+info from parent ";
            //Plane Seems fully specified
            //all T2 plane attribute read correctly. Check if it is consitent
            unsigned int test_position_t2=arm*20+ht*10+pl*2+pls;
            unsigned int testHS=pl*2+pls;
            if(testHS!=position_t2) {
              edm::LogPrint("TotemDAQMappingESSourceXML") <<"T2 Xml inconsistence in pl-pls attributes and position. Only 'position attribute' taken ";
            }

            // For plane, ID_t2 should go from 0..39 position_t2 from 0..9
            ID_t2=parentID+position_t2;
            cout << "attribcounter_t2planedescript>=(25), ID_t2: " << ID_t2 << endl;
            toaddForParentID=position_t2;
            if (ID_t2!=(int)test_position_t2)
              edm::LogPrint("TotemDAQMappingESSourceXML") <<"T2 Xml inconsistence in plane attributes and xml parents structure. Plane attributes ignored";

          } else {
            // Case where arm-ht-pl-pls are NOT specified
            edm::LogVerbatim("TotemDAQMappingESSourceXML")<<"T2 Plane have parentID: "<<parentID<<" for its VFATs. Plane Position read: "<<position_t2;

            if (attribcounter_t2planedescript==21) {
              // You have put in XML only position and not Plane id (the last is obligatory)
              ID_t2=parentID+position_t2;
              // cout << "attribcounter_t2planedescript>=(21), ID_t2: " << ID_t2 << endl;
              toaddForParentID=position_t2;
              idSet_t2=true;
            }
          }
        } else {
          // Construct plane position from other attributes cause "position" is not inserted;
          // Ex- monitor:    <t2_detector id="0" z="13871.3" arm="0" ht="0" pl="0" pls="0" >

          if (attribcounter_t2planedescript>=1) {
            // Remember, Z attribute is not counted

            if(attribcounter_t2planedescript>=5) {
              int test_position_t2=arm*20+ht*10+pl*2+pls;

              // case for xml from monitor
              ID_t2=test_position_t2;
              cout << "ID_t2=test_position_t2: " << ID_t2 << endl;
              toaddForParentID=test_position_t2;

              if ((int)parentID!=ID_t2) {
                edm::LogPrint("TotemDAQMappingESSourceXML") <<"T2 Inconsistence between plane 'id' and position from attributes. Id ignored";
                edm::LogPrint("TotemDAQMappingESSourceXML") <<" T2-Parent = "<<parentID;
              }
            } else {
              toaddForParentID=ID_t2;
              edm::LogVerbatim("TotemDAQMappingESSourceXML")<<" Number of T2 plane attributes: "<< attribcounter_t2planedescript<<" T2-Plane attribute utilezed for parentID: plane 'id' only";
            }
          } else {
//            position_t2Set = false;
            edm::LogProblem ("TotemDAQMappingESSourceXML") << "T2 plane not enough specified from its attribute!";
          }
        }
      }

      // content control
      if (idSet_t2 == false) {
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "ID_t2 not given for element `" << XMLString::transcode(n->getNodeName()) << "'" << endl;
        edm::LogProblem ("TotemDAQMappingESSourceXML") <<"ID_t2 not given for element `"<<XMLString::transcode(n->getNodeName()) << "'";
      }

      if (type == nChip && !hw_id_set)
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "hw_id not given for VFAT id `" <<
          ID_t2 << "'" << endl;

      if (type == nT2Det && position_t2 > 39) {
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "Plane position_t2 range from 0 to 39. position_t2 = " << position_t2 << " is invalid." << endl;
        edm::LogProblem ("TotemDAQMappingESSourceXML") <<"Plane position_t2 range from 0 to 39. position_t2 = "<<position_t2<< " is invalid.";
      }
    }


    if (type == nChip) {
      // save mapping data
      if (pType == pMapping) {
#ifdef DEBUG
        printf("T2 Vfat in plane (parentID): %i || GeomPosition %i \n", parentID, ID_t2);
        printf("\t\t\tID_t2 = 0x%x\n", hw_id);
        printf("\t\t\tpos = %i\n", position_t2);
#endif     
        unsigned int symId=0;
        // Check if it is a special chip
        if (!tagT2detector.compare(XMLString::transcode((n->getParentNode()->getNodeName()))))
          symId = parentID * 100 + ID_t2; // same conv = symbplaneNumber*100 +iid used in DQM
        else {
          // It is a special VFAT and the special number is set directly in the XML file
          symId = ID_t2;      //17,18,19,20
#ifdef DEBUG
          printf("TotemDAQMappingESSourceXML Found T2 special Vfat ChId-SLink-Symb  0x%x - %i - %i \n",
              ID_t2,position_t2,symId );
#endif
        }

        TotemFramePosition framepos = ChipFramePosition(n);
        TotemVFATInfo vfatInfo;
        vfatInfo.symbolicID.symbolicID = symId;
        vfatInfo.hwID = hw_id;
        vfatInfo.symbolicID.subSystem = TotemSymbID::T2;
        vfatInfo.type = TotemVFATInfo::data;
        data->insert(framepos, vfatInfo);
      }

      // save mask data
      if (pType == pMask) {
        TotemVFATAnalysisMask vfatMask;
        TotemSymbID symbId;
        symbId.subSystem = TotemSymbID::T2;
        symbId.symbolicID = 100*parentID+ID_t2;

        DOMNode *fullMaskNode = attr->getNamedItem(XMLString::transcode("full_mask"));
        if (fullMaskNode && !GetNodeValue(fullMaskNode).compare("yes"))
          vfatMask.fullMask = true;
        else
          GetChannels(n, vfatMask.maskedChannels);

        mask->insert(symbId, vfatMask);
        //cout << "saved mask, ID = " << symbId.symbolicID << ", full mask: " << vfatMask.fullMask << endl;
      }
    } else {
      // Look for the children of n (recursion)
      // 3° argument=parentId  is needed for calculate VFAT-id startintg from the parent plane
      ParseTreeT2(pType, n, type, parentID+toaddForParentID, data, mask);
    }
  } // Go to the next children
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::ParseTreeT1(ParseType pType, xercesc::DOMNode * parent, NodeType parentType,
  unsigned int parentID, const boost::shared_ptr<TotemDAQMapping>& mapping,
  const boost::shared_ptr<TotemAnalysisMask>& mask, unsigned int T1Arm, unsigned int T1Plane, unsigned int T1CSC)
{
  const int ArmMask = 0x0200;
  const int PlaneMask = 0x01c0;
  const int CSCMask = 0x0038;
  const int GenderMask = 0x0004;
  const int VFnMask = 0x0003;

  int ArmMask_ = 0;
  int PlaneMask_ = 0;
  int CSCMask_ = 0;
  int GenderMask_ = 0;
  int VFnMask_ = 0;

  unsigned int T1ChannelType = 0;
  
  DOMNodeList *children = parent->getChildNodes();

#ifdef DEBUG
  printf(">> ParseTreeT1(parent,parentType,parentID)=(%p, %i, %u)\n", parent, parentType, parentID);
  printf("\tchildren: Numero children: %li\n", children->getLength());
#endif

  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode *n = children->item(i);

    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    // get node type for RP or T2 or T1
    NodeType type = GetNodeType(n);

#ifdef DEBUG
    printf("\t\tchildren #%i: is a %s, (of type %i) \n", i, XMLString::transcode(n->getNodeName()), type);
#endif

    if ((type == nUnknown)) {
#ifdef DEBUG
      printf("Found Unknown tag during T1 reading.. EXIT ");
#endif   
      continue;
    }

    if ((T1Node(type)==false)&&(CommonNode(type)==false)) {
#ifdef DEBUG
      printf("Found Non-T1 tag during T1 reading.. EXIT ");
      printf("\t The tag is:  %s \n", XMLString::transcode(n->getNodeName()));
#endif
      continue;
    }

    // id  for T2 plane goes from 0..9; for chip is the 16 bit ID
    // for VFATs: (0 or 1 for anodes A0 and A1; 0, 1 or 2 for Cathodes C0,C1,C2)
    unsigned int ID_t1 = 0;
#ifdef DEBUG
    unsigned int VFPOS = 0;
#endif
    unsigned int Gender =0;

    // hardware of an id
    unsigned int hw_id = 0;
    bool hw_id_set = false;

    bool idSet_t1 = false;//bool position_t2Set = false;

    DOMNamedNodeMap* attr = n->getAttributes();

	bool fullMask = false;
    
    //    Begin loop for save T1 element attriute  ------------------------------------------------------------------
    for (unsigned int j = 0; j < attr->getLength(); j++) {

      DOMNode *a = attr->item(j);
      if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
        sscanf(XMLString::transcode(a->getNodeValue()), "%u", &ID_t1);
        idSet_t1 = true;
      }

      if (type == nT1Arm) {
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          // arm is the top node and parent ID should be reset to 0.
          parentID=0;
          unsigned int id_arm = atoi(XMLString::transcode(a->getNodeValue()));
          T1Arm = id_arm;
          if (id_arm != 0 && id_arm != 1) {
            throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "T1 id_arm neither 0 nor 1. Problem parsing XML file." << XMLString::transcode(n->getNodeName()) << endl;
            edm::LogProblem ("TotemDAQMappingESSourceXML") <<"T1 id_arm neither 0 nor 1. Problem parsing XML file."<<XMLString::transcode(n->getNodeName()) ;
          }

          id_arm = id_arm << 9;
          ArmMask_ = ~ArmMask;
          parentID &= ArmMask_;
          parentID |= id_arm;
        }
      }

      if (type == nT1Plane) {
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          unsigned int id_plane = atoi(XMLString::transcode(a->getNodeValue()));
          T1Plane = id_plane;
          id_plane = id_plane << 6;
          PlaneMask_ = ~PlaneMask;
          parentID &= PlaneMask_;
          parentID |= id_plane;
        }
      }

      if (type == nT1CSC) {
        if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
          unsigned int id_csc = atoi(XMLString::transcode(a->getNodeValue()));
          T1CSC = id_csc;
          id_csc = id_csc << 3;
          CSCMask_ = ~CSCMask;
          parentID &= CSCMask_;
          parentID |= id_csc;
        }
      }

      if (type == nT1ChannelType) {
		if (!strcmp(XMLString::transcode(a->getNodeName()), "id")) {
		  T1ChannelType = atoi(XMLString::transcode(a->getNodeValue()));
		}
		if (!strcmp(XMLString::transcode(a->getNodeName()), "full_mask")) {
		  fullMask = (strcmp(XMLString::transcode(a->getNodeValue()), "no") != 0);
		}
      }
      
      if(type == nChip) {
#ifdef DEBUG
        if (!strcmp(XMLString::transcode(a->getNodeName()), "position")){
          VFPOS = atoi(XMLString::transcode(a->getNodeValue()));
        }
#endif
        if (!strcmp(XMLString::transcode(a->getNodeName()), "polarity")) {
          if (!strcmp(XMLString::transcode(a->getNodeValue()),"a")) {
            Gender = 0;
          } else
            if (!strcmp(XMLString::transcode(a->getNodeValue()),"c")) {
              Gender = 1;
            } else {
              throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "T1: Neither anode nor cathode vfat : " << XMLString::transcode(n->getNodeName()) << endl;
              edm::LogProblem ("TotemDAQMappingESSourceXML") <<"T1: Neither anode nor cathode vfat : "<<XMLString::transcode(n->getNodeName());
            }
        }

        if (!strcmp(XMLString::transcode(a->getNodeName()), "hw_id"))
          sscanf(XMLString::transcode(a->getNodeValue()), "%x", &hw_id);
          hw_id_set = true;
      }
    }

    // content control
    // Note: each element has an id!! However if the element is a plane, it could be enough to use position 0..9

    if (idSet_t1==false){
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "ID_t1 not given for element `" << XMLString::transcode(n->getNodeName()) << "'" << endl;
      edm::LogProblem ("TotemDAQMappingESSourceXML") <<"ID_t1 not given for element `"<<XMLString::transcode(n->getNodeName()) << "'";
    }

    if (type == nChip && !hw_id_set)
      throw cms::Exception("TotemDAQMappingESSourceXML::ParseTreeT1") <<
        "hw_id not set for T1 VFAT id " << ID_t1 << "." << endl;



			// save mask data
	if (type == nT1ChannelType && pType == pMask) {
		TotemSymbID symbId;
		symbId.subSystem = TotemSymbID::T1;
		symbId.symbolicID = T1ChannelType + 10 * T1CSC + 100 * T1Plane + 1000 * T1Arm;
		//cout << "mask: " << T1Arm << " " << T1Plane << " " << T1CSC <<  " " <<T1ChannelType << endl;
		TotemVFATAnalysisMask am;
		am.fullMask = fullMask;
		GetChannels(n, am.maskedChannels);
		mask->insert(symbId, am);
		//cout << "saved mask, ID = " << symbId.symbolicID << ", full mask: " << am.fullMask << endl;
	}
    
    // save data
    if (type == nChip) {
#ifdef DEBUG
      printf("T1 Vfat in detector (parentID): %x || Position %i \n", parentID, VFPOS);
      printf("\t\t\tID_t1 = 0x%x\n", ID_t1);
#endif     

      unsigned int symId=0;

      // Check if it is a special chip
      if (!tagT1CSC.compare(XMLString::transcode((n->getParentNode()->getNodeName())))) {
        symId = parentID;
        Gender = Gender << 2;
        GenderMask_ = ~GenderMask;

        symId &= GenderMask_;
        symId |= Gender;

        VFnMask_ = ~VFnMask;
        symId &= VFnMask_;
        symId |= ID_t1;
      } else {
        // It is a special VFAT ...
        throw cms::Exception("TotemDAQMappingESSourceXML::ParseTree") << "T1 has no special vfat `" << XMLString::transcode(n->getNodeName()) << "'" << endl;
        edm::LogProblem ("TotemDAQMappingESSourceXML") <<"T1 has no special vfat `"<<XMLString::transcode(n->getNodeName()) << "'";
      }

      // Assign a contanaier for the register of that VFAT with ChipId (Hex)
      TotemFramePosition framepos = ChipFramePosition(n);
      TotemVFATInfo vfatInfo;
      vfatInfo.symbolicID.symbolicID = symId;
      vfatInfo.hwID = hw_id;
      vfatInfo.symbolicID.subSystem = TotemSymbID::T1;
      vfatInfo.type = TotemVFATInfo::data;
      mapping->insert(framepos, vfatInfo);
    } else {
      // Look for the children of n (recursion)
      // 3° argument=parentId  is needed for calculate VFAT-id startintg from the parent plane
      ParseTreeT1(pType, n, type, parentID, mapping, mask, T1Arm, T1Plane, T1CSC);
    }
  } // Go to the next children
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemFramePosition TotemDAQMappingESSourceXML::ChipFramePosition(xercesc::DOMNode *chipnode)
{
  TotemFramePosition fp;
  unsigned char attributeFlag = 0;

  DOMNamedNodeMap* attr = chipnode->getAttributes();
  for (unsigned int j = 0; j < attr->getLength(); j++)
  {
    DOMNode *a = attr->item(j);
    if (fp.setXMLAttribute(XMLString::transcode(a->getNodeName()), XMLString::transcode(a->getNodeValue()), attributeFlag) > 1)
    {
      throw cms::Exception("TotemDAQMappingESSourceXML") <<
        "Unrecognized tag `" << XMLString::transcode(a->getNodeName()) <<
        "' or incompatible value `" << XMLString::transcode(a->getNodeValue()) <<
        "'." << endl;
    }
  }

  if (!fp.checkXMLAttributeFlag(attributeFlag))
  {
    throw cms::Exception("TotemDAQMappingESSourceXML") <<
      "Wrong/incomplete DAQ channel specification (attributeFlag = " << attributeFlag << ")." << endl;
  }

  return fp;
}

//----------------------------------------------------------------------------------------------------

TotemDAQMappingESSourceXML::NodeType TotemDAQMappingESSourceXML::GetNodeType(xercesc::DOMNode *n)
{
  // common node types
  if (Test(n, tagArm)) return nArm;
  if (Test(n, tagChip1)) return nChip;
  if (Test(n, tagChip2)) return nChip;
  if (Test(n, tagTriggerVFAT1)) return nTriggerVFAT;

  // RP node types
  if (Test(n, tagRPStation)) return nRPStation;
  if (Test(n, tagRPPot)) return nRPPot;
  if (Test(n, tagRPPlane)) return nRPPlane;

  // T2 node types
  if (Test(n, tagT2)) return nT2;
  if (Test(n, tagT2detector)) return nT2Det;
  if (Test(n, tagT2Half)) return nT2Half;

  // T1 node types
  if (Test(n, tagT1)) return nT1;
  if (Test(n, tagT1Arm)) return nT1Arm;
  if (Test(n, tagT1Plane)) return nT1Plane;
  if (Test(n, tagT1CSC)) return nT1CSC;
  if (Test(n, tagT1ChannelType)) return nT1ChannelType;
  if (Test(n, tagChannel)) return nChannel;


  throw cms::Exception("TotemDAQMappingESSourceXML::GetNodeType") << "Unknown tag `"
    << XMLString::transcode(n->getNodeName()) << "'.\n";
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMappingESSourceXML::GetChannels(xercesc::DOMNode *n, set<unsigned char> &channels)
{
  DOMNodeList *children = n->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++)
  {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE || !Test(n, "channel"))
      continue;

    DOMNamedNodeMap* attr = n->getAttributes();
    bool idSet = false;
    for (unsigned int j = 0; j < attr->getLength(); j++)
    {
      DOMNode *a = attr->item(j);

      if (!strcmp(XMLString::transcode(a->getNodeName()), "id"))
      {
        unsigned int id = 0;
        sscanf(XMLString::transcode(a->getNodeValue()), "%u", &id);
        channels.insert(id);
        idSet = true;
        break;
      }
    }

    if (!idSet)
    {
      throw cms::Exception("TotemDAQMappingESSourceXML::GetChannels") <<
        "Channel tags must have an `id' attribute.";
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(TotemDAQMappingESSourceXML);
