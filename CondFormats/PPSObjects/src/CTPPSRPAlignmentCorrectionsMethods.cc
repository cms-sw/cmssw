/****************************************************************************
 *
 * This is a part of CMS-TOTEM PPS offline software.
 * Authors:
 * Jan Kaspar
 * Helena Malbouisson
 * Clemencia Mora Herrera
 *
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "Utilities/Xerces/interface/XercesStrUtils.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <map>
#include <set>

#ifdef XERCES_CPP_NAMESPACE_USE
XERCES_CPP_NAMESPACE_USE
#endif

//----------------------------------------------------------------------------------------------------

/**
STRUCTURE OF CTPPS ALINGMENT XML FILE
The file has the following structure
<code>
<xml>
  <iov first="run:ls" last="run:ls">
    <tag/>
    <tag/>
    ...
  </iov>
  <iov first="run:ls" last="run:ls">
    ...
  </iov>
  .
  .
  .
</xml>
</code>
The time intervals are specified by the `first' and `last' run-lumisection pairs.
If the <iov> tag is not present, an infinite validty is assumed for all the tags.
The tag can be either
* "det" - the alignment correction is applied to one detector or
* "rp" - the alignment correction id applied to one RP
Each tag must have an "id" attribute set. In addition the following attributes are recognized:
* sh_x - shift in x
* sh_x_e - the uncertainty of sh_x determination
* sh_y - shift in y
* sh_y_e - the uncertainty of sh_y determination
* sh_z - shift in z
* sh_z_e - the uncertainty of sh_z determination
* rot_x - rotation around x
* rot_x_e - the uncertainty of rot_x determination
* rot_y - rotation around y
* rot_y_e - the uncertainty of rot_y determination
* rot_z - rotation around z
* rot_z_e - the uncertainty of rot_z determination
UNITS: shifts are in um, rotations are in mrad.
*/

//----------------------------------------------------------------------------------------------------
edm::IOVSyncValue CTPPSRPAlignmentCorrectionsMethods::stringToIOVValue(const std::string& str) {
  if (str == "-inf")
    return edm::IOVSyncValue::beginOfTime();

  if (str == "+inf")
    return edm::IOVSyncValue::endOfTime();

  size_t sep_pos = str.find(':');
  const std::string& runStr = str.substr(0, sep_pos);
  const std::string& lsStr = str.substr(sep_pos + 1);

  return edm::IOVSyncValue(edm::EventID(atoi(runStr.c_str()), atoi(lsStr.c_str()), 1));
}

//----------------------------------------------------------------------------------------------------

std::string CTPPSRPAlignmentCorrectionsMethods::iovValueToString(const edm::IOVSyncValue& val) {
  if (val == edm::IOVSyncValue::beginOfTime())
    return "-inf";

  if (val == edm::IOVSyncValue::endOfTime())
    return "+inf";

  char buf[50];
  sprintf(buf, "%u:%u", val.eventID().run(), val.eventID().luminosityBlock());
  return buf;
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataSequence CTPPSRPAlignmentCorrectionsMethods::loadFromXML(const std::string& fileName) {
  // prepare output
  CTPPSRPAlignmentCorrectionsDataSequence output;

  // load DOM tree
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
        << "An XMLException caught with message: " << cms::xerces::toString(toCatch.getMessage()) << ".";
  }

  auto parser = std::make_unique<XercesDOMParser>();
  parser->setValidationScheme(XercesDOMParser::Val_Always);
  parser->setDoNamespaces(true);
  parser->parse(fileName.c_str());

  if (!parser)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
        << "Cannot parse file `" << fileName << "' (parser = NULL).";

  DOMDocument* xmlDoc = parser->getDocument();

  if (!xmlDoc)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
        << "Cannot parse file `" << fileName << "' (xmlDoc = NULL).";

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  if (!elementRoot)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods") << "File `" << fileName << "' is empty.";

  // extract useful information form the DOM tree
  DOMNodeList* children = elementRoot->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode* node = children->item(i);
    if (node->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    const std::string node_name = cms::xerces::toString(node->getNodeName());

    // check node type
    unsigned char nodeType = 0;
    if (node_name == "iov")
      nodeType = 1;
    else if (node_name == "det")
      nodeType = 2;
    else if (node_name == "rp")
      nodeType = 3;

    if (nodeType == 0)
      throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods") << "Unknown node `" << node_name << "'.";

    // for backward compatibility: support files with no iov block
    if (nodeType == 2 || nodeType == 3) {
      const edm::ValidityInterval iov(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
      output.insert(iov, getCorrectionsData(elementRoot));
      break;
    }

    // get attributes
    edm::IOVSyncValue first, last;
    bool first_set = false, last_set = false;
    DOMNamedNodeMap* attrs = node->getAttributes();
    for (unsigned int j = 0; j < attrs->getLength(); j++) {
      const DOMNode* attr = attrs->item(j);
      const std::string attr_name = cms::xerces::toString(attr->getNodeName());

      if (attr_name == "first") {
        first_set = true;
        first = stringToIOVValue(cms::xerces::toString(attr->getNodeValue()));
      } else if (attr_name == "last") {
        last_set = true;
        last = stringToIOVValue(cms::xerces::toString(attr->getNodeValue()));
      } else
        edm::LogProblem("CTPPSRPAlignmentCorrectionsMethods")
            << ">> CTPPSRPAlignmentCorrectionsDataSequence::loadFromXML > Warning: unknown attribute `" << attr_name
            << "'.";
    }

    // interval of validity must be set
    if (!first_set || !last_set)
      throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
          << "iov tag must have `first' and `last' attributes set.";

    // process data
    CTPPSRPAlignmentCorrectionsData corrections = CTPPSRPAlignmentCorrectionsMethods::getCorrectionsData(node);

    // save result
    output.insert(edm::ValidityInterval(first, last), corrections);
  }

  // clean up
  parser.reset();
  XMLPlatformUtils::Terminate();

  return output;
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsData CTPPSRPAlignmentCorrectionsMethods::getCorrectionsData(DOMNode* root) {
  CTPPSRPAlignmentCorrectionsData result;

  DOMNodeList* children = root->getChildNodes();
  for (unsigned int i = 0; i < children->getLength(); i++) {
    DOMNode* node = children->item(i);
    if (node->getNodeType() != DOMNode::ELEMENT_NODE)
      continue;

    const std::string node_name = cms::xerces::toString(node->getNodeName());

    // check node type
    unsigned char nodeType = 0;
    if (node_name == "det")
      nodeType = 1;
    else if (node_name == "rp")
      nodeType = 2;

    if (nodeType == 0)
      throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
          << "Unknown node `" << cms::xerces::toString(node->getNodeName()) << "'.";

    // check children
    if (node->getChildNodes()->getLength() > 0) {
      edm::LogProblem("CTPPSRPAlignmentCorrectionsMethods")
          << "LoadXMLFile > Warning: tag `" << cms::xerces::toString(node->getNodeName()) << "' has "
          << node->getChildNodes()->getLength() << " children nodes - they will be all ignored.";
    }

    // default values
    double sh_x = 0., sh_y = 0., sh_z = 0., rot_x = 0., rot_y = 0., rot_z = 0.;
    double sh_x_e = 0., sh_y_e = 0., sh_z_e = 0., rot_x_e = 0., rot_y_e = 0., rot_z_e = 0.;
    unsigned int id = 0;
    bool idSet = false;

    // get attributes
    DOMNamedNodeMap* attr = node->getAttributes();
    for (unsigned int j = 0; j < attr->getLength(); j++) {
      DOMNode* a = attr->item(j);
      const std::string node_name = cms::xerces::toString(a->getNodeName());

      if (node_name == "id") {
        id = cms::xerces::toUInt(a->getNodeValue());
        idSet = true;
      } else if (node_name == "sh_x")
        sh_x = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "sh_x_e")
        sh_x_e = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "sh_y")
        sh_y = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "sh_y_e")
        sh_y_e = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "sh_z")
        sh_z = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "sh_z_e")
        sh_z_e = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_x")
        rot_x = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_x_e")
        rot_x_e = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_y")
        rot_y = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_y_e")
        rot_y_e = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_z")
        rot_z = cms::xerces::toDouble(a->getNodeValue());
      else if (node_name == "rot_z_e")
        rot_z_e = cms::xerces::toDouble(a->getNodeValue());
      else
        edm::LogProblem("CTPPSRPAlignmentCorrectionsMethods")
            << ">> CTPPSRPAlignmentCorrectionsMethods::getCorrectionsData > Warning: unknown attribute `"
            << cms::xerces::toString(a->getNodeName()) << "'.";
    }

    // id must be set
    if (!idSet)
      throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
          << "Id not set for tag `" << cms::xerces::toString(node->getNodeName()) << "'.";

    // build alignment
    const CTPPSRPAlignmentCorrectionData align_corr(sh_x * 1e-3,
                                                    sh_x_e * 1e-3,
                                                    sh_y * 1e-3,
                                                    sh_y_e * 1e-3,
                                                    sh_z * 1e-3,
                                                    sh_z_e * 1e-3,
                                                    rot_x * 1e-3,
                                                    rot_x_e * 1e-3,
                                                    rot_y * 1e-3,
                                                    rot_y_e * 1e-3,
                                                    rot_z * 1e-3,
                                                    rot_z_e * 1e-3);

    // add the alignment to the right list
    if (nodeType == 1)
      result.addSensorCorrection(id, align_corr, true);
    if (nodeType == 2)
      result.addRPCorrection(id, align_corr, true);
  }

  return result;
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsMethods::writeToXML(const CTPPSRPAlignmentCorrectionsDataSequence& data,
                                                    const std::string& fileName,
                                                    bool precise,
                                                    bool wrErrors,
                                                    bool wrSh_xy,
                                                    bool wrSh_z,
                                                    bool wrRot_xy,
                                                    bool wrRot_z) {
  FILE* rf = fopen(fileName.c_str(), "w");
  if (!rf)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsMethods")
        << "Cannot open file `" << fileName << "' to save alignments.";

  fprintf(rf, "<!-- Shifts in um, rotations in mrad. -->\n");
  fprintf(rf, "<xml DocumentType=\"AlignmentDescription\">\n");

  // write all IOVs
  for (const auto& p : data) {
    fprintf(rf,
            "\t<iov first=\"%s\" last=\"%s\">\n",
            iovValueToString(p.first.first()).c_str(),
            iovValueToString(p.first.last()).c_str());

    writeXMLBlock(p.second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);

    fprintf(rf, "\t</iov>\n");
  }

  fprintf(rf, "</xml>\n");
  fclose(rf);
}

void CTPPSRPAlignmentCorrectionsMethods::writeXMLBlock(const CTPPSRPAlignmentCorrectionsData& data,
                                                       FILE* rf,
                                                       bool precise,
                                                       bool wrErrors,
                                                       bool wrSh_xy,
                                                       bool wrSh_z,
                                                       bool wrRot_xy,
                                                       bool wrRot_z) {
  bool firstRP = true;
  unsigned int prevRP = 0;
  std::set<unsigned int> writtenRPs;

  const auto& sensors = data.getSensorMap();
  const auto& rps = data.getRPMap();

  for (auto it = sensors.begin(); it != sensors.end(); ++it) {
    CTPPSDetId sensorId(it->first);
    unsigned int rpId = sensorId.rpId();
    unsigned int decRPId = sensorId.arm() * 100 + sensorId.station() * 10 + sensorId.rp();

    // start a RP block
    if (firstRP || prevRP != rpId) {
      if (!firstRP)
        fprintf(rf, "\n");
      firstRP = false;

      fprintf(rf, "\t<!-- RP %3u -->\n", decRPId);

      auto rit = rps.find(rpId);
      if (rit != rps.end()) {
        fprintf(rf, "\t<rp id=\"%u\"                  ", rit->first);
        writeXML(rit->second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
        fprintf(rf, "/>\n");
        writtenRPs.insert(rpId);
      }
    }
    prevRP = rpId;

    // write plane id
    unsigned int planeIdx = 1000;
    if (sensorId.subdetId() == CTPPSDetId::sdTrackingStrip)
      planeIdx = TotemRPDetId(it->first).plane();
    if (sensorId.subdetId() == CTPPSDetId::sdTrackingPixel)
      planeIdx = CTPPSPixelDetId(it->first).plane();
    if (sensorId.subdetId() == CTPPSDetId::sdTimingDiamond)
      planeIdx = CTPPSDiamondDetId(it->first).plane();
    fprintf(rf, "\t<!-- plane %u --> ", planeIdx);

    // write the correction
    fprintf(rf, "<det id=\"%u\"", it->first);
    writeXML(it->second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
    fprintf(rf, "/>\n");
  }

  // write remaining RPs
  for (auto it = rps.begin(); it != rps.end(); ++it) {
    std::set<unsigned int>::iterator wit = writtenRPs.find(it->first);
    if (wit == writtenRPs.end()) {
      CTPPSDetId rpId(it->first);
      unsigned int decRPId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

      if (!firstRP)
        fprintf(rf, "\n");
      firstRP = false;

      fprintf(rf, "\t<!-- RP %3u -->\n", decRPId);

      fprintf(rf, "\t<rp id=\"%u\"                  ", it->first);
      writeXML(it->second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
      fprintf(rf, "/>\n");
    }
  }
}

//----------------------------------------------------------------------------------------------------

#define WRITE(q, tag, dig, lim)                 \
  if (precise)                                  \
    fprintf(f, " " tag "=\"%.15E\"", q * 1E3);  \
  else if (fabs(q * 1E3) < lim && q != 0)       \
    fprintf(f, " " tag "=\"%+8.1E\"", q * 1E3); \
  else                                          \
    fprintf(f, " " tag "=\"%+8." #dig "f\"", q * 1E3);

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsMethods::writeXML(const CTPPSRPAlignmentCorrectionData& data,
                                                  FILE* f,
                                                  bool precise,
                                                  bool wrErrors,
                                                  bool wrSh_xy,
                                                  bool wrSh_z,
                                                  bool wrRot_xy,
                                                  bool wrRot_z) {
  if (wrSh_xy) {
    WRITE(data.getShX(), "sh_x", 2, 0.1);
    WRITE(data.getShY(), "sh_y", 2, 0.1);
    if (wrErrors) {
      WRITE(data.getShXUnc(), "sh_x_e", 2, 0.1);
      WRITE(data.getShYUnc(), "sh_y_e", 2, 0.1);
    }
  }

  if (wrSh_z) {
    WRITE(data.getShZ(), "sh_z", 2, 0.1);
    if (wrErrors) {
      WRITE(data.getShZUnc(), "sh_z_e", 2, 0.1);
    }
  }

  if (wrRot_xy) {
    WRITE(data.getRotX(), "rot_x", 3, 0.01);
    WRITE(data.getRotY(), "rot_y", 3, 0.01);
    if (wrErrors) {
      WRITE(data.getRotXUnc(), "rot_x_e", 3, 0.01);
      WRITE(data.getRotYUnc(), "rot_y_e", 3, 0.01);
    }
  }

  if (wrRot_z) {
    WRITE(data.getRotZ(), "rot_z", 3, 0.01);
    if (wrErrors) {
      WRITE(data.getRotZUnc(), "rot_z_e", 3, 0.01);
    }
  }
}
