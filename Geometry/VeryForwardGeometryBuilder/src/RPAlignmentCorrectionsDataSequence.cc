/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsDataSequence.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

using namespace std;
using namespace edm;
using namespace xercesc;


/**
STRUCTURE OF ALINGMENT XML FILE

The file has the following structure
<code>
<xml>
  <TimeInterval first="..." last="...">
    <tag/>
    <tag/>
    ...
  </TimeInterval>
  <TimeInterval first="..." last="...">
    ...
  </TimeInterval>
  .
  .
  .
</xml>
</code>

The time intervals are specified by the `first' and `last' UNIX timestamp (boundaries included).
If there is only one time interval, the <TimeInterval> tags might be omitted. An infinite validty
is assumed in this case.

The tag can be either
  * "det" - the alignment correction is applied to one detector or
  * "rp" - the alignment correction id applied to one RP

Each tag must have an "id" attribute set. In addition the following attributes are recognized:
  * sh_r - shift in readout direction
  * sh_r_e - the uncertainty of sh_r determination
  * sh_x - shift in x
  * sh_x_e - the uncertainty of sh_x determination
  * sh_y - shift in y
  * sh_y_e - the uncertainty of sh_y determination
  * sh_z - shift in z
  * sh_z_e - the uncertainty of sh_z determination
  * rot_z - rotation around z
  * rot_z_e - the uncertainty of rot_z determination

UNITS: shifts are in um, rotations are in mrad.
 */

void RPAlignmentCorrectionsDataSequence::LoadXMLFile(const string &fileName)
{
  // prepend CMSSW src dir
  char *cmsswPath = getenv("CMSSW_BASE");
  size_t start = fileName.find_first_not_of("   ");
  string fn = fileName.substr(start);
  if (cmsswPath && fn[0] != '/' && fn.find("./") != 0)
    fn = string(cmsswPath) + string("/src/") + fn;

  // load DOM tree first the file
  try
  {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch)
  {
    char* message = XMLString::transcode(toCatch.getMessage());
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "An XMLException caught with message: " << message << ".\n";
    XMLString::release(&message);
  }

  XercesDOMParser* parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);
  parser->setDoNamespaces(true);

  try
  {
    parser->parse(fn.c_str());
  }
  catch (...)
  {
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Cannot parse file `" << fn << "' (exception)." << endl;
  }

  if (!parser)
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Cannot parse file `" << fn << "' (parser = NULL)." << endl;
  
  DOMDocument* xmlDoc = parser->getDocument();

  if (!xmlDoc)
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Cannot parse file `" << fn << "' (xmlDoc = NULL)." << endl;

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  if (!elementRoot)
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "File `" << fn << "' is empty." << endl;
  
  // extract useful information form the DOM tree
  DOMNodeList *children = elementRoot->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++)
  {
    DOMNode *n = children->item(i);
    if (n->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;
   
    // check node type
    unsigned char nodeType = 0;
    if (!strcmp(XMLString::transcode(n->getNodeName()), "TimeInterval")) nodeType = 1;
    if (!strcmp(XMLString::transcode(n->getNodeName()), "det")) nodeType = 2;
    if (!strcmp(XMLString::transcode(n->getNodeName()), "rp")) nodeType = 3;

    if (!nodeType)
      throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Unknown node `" << XMLString::transcode(n->getNodeName()) << "'.";

    // old style - no TimeInterval block?
    if (nodeType == 2 || nodeType == 3)
    {
      //printf(">> RPAlignmentCorrectionsDataSequence::LoadXMLFile > WARNING:\n\tIn file `%s' no TimeInterval given, assuming one block of infinite validity.\n", fileName.c_str());

      TimeValidityInterval inf;
      inf.SetInfinite();
      insert(pair<TimeValidityInterval, RPAlignmentCorrectionsData>(inf, RPAlignmentCorrectionsMethods::GetCorrectionsData(elementRoot)));
      break;
    }

    // get attributes
    TimeValue_t first=0, last=0;
    bool first_set = false, last_set = false;
    DOMNamedNodeMap* attr = n->getAttributes();
    for (unsigned int j = 0; j < attr->getLength(); j++)
    {    
      DOMNode *a = attr->item(j);
 
      if (!strcmp(XMLString::transcode(a->getNodeName()), "first"))
      {
        first_set = true;
        first = TimeValidityInterval::UNIXStringToValue(XMLString::transcode(a->getNodeValue()));
      } else if (!strcmp(XMLString::transcode(a->getNodeName()), "last"))
      {
        last_set = true;
        last = TimeValidityInterval::UNIXStringToValue(XMLString::transcode(a->getNodeValue()));
      } else
        edm::LogProblem("RPAlignmentCorrectionsDataSequence") << ">> RPAlignmentCorrectionsDataSequence::LoadXMLFile > Warning: unknown attribute `"
          << XMLString::transcode(a->getNodeName()) << "'.";
    }

    // interval of validity must be set
    if (!first_set || !last_set)
      throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "TimeInterval tag must have `first' and `last' attributes set.";

    TimeValidityInterval tvi(first, last);
    
    // process data
    RPAlignmentCorrectionsData corrections = RPAlignmentCorrectionsMethods::GetCorrectionsData(n);

    // save result
    insert(pair<TimeValidityInterval, RPAlignmentCorrectionsData>(tvi, corrections));
  }

  XMLPlatformUtils::Terminate();
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionsDataSequence::WriteXMLFile(const string &fileName, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy,
  bool wrSh_z, bool wrRot_z) const
{
  FILE *rf = fopen(fileName.c_str(), "w");
  if (!rf)
    throw cms::Exception("RPAlignmentCorrectionsDataSequence::WriteXMLFile") << "Cannot open file `" << fileName
      << "' to save alignments." << endl;

  fprintf(rf, "<!--\nShifts in um, rotations in mrad.\n\nFor more details see RPAlignmentCorrections::LoadXMLFile in\n");
  fprintf(rf, "Alignment/RPDataFormats/src/RPAlignmentCorrectionsDataSequence.cc\n-->\n\n");
  fprintf(rf, "<xml DocumentType=\"AlignmentSequenceDescription\">\n");

  // write all time blocks
  for (const_iterator it = this->begin(); it != this->end(); ++it)
  {
    fprintf(rf, "\t<TimeInterval first=\"%s\" last=\"%s\">",
      TimeValidityInterval::ValueToUNIXString(it->first.first).c_str(),
      TimeValidityInterval::ValueToUNIXString(it->first.last).c_str()
    );

    RPAlignmentCorrectionsMethods::WriteXMLBlock(it->second, rf, precise, wrErrors, wrSh_r, wrSh_xy, wrSh_z, wrRot_z );
    fprintf(rf, "\t</TimeInterval>\n");
  }

  fprintf(rf, "</xml>\n");
  fclose(rf);
}

