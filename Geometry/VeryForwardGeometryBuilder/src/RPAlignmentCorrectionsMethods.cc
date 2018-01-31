/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsMethods.h"

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

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionsData
RPAlignmentCorrectionsMethods::getCorrectionsDataFromFile( const edm::FileInPath& fileName )
{
  edm::LogInfo("RPAlignmentCorrectionsMethods")
    << "LoadXMLFile(" << fileName << ")";

  // load DOM tree first the file
  XMLPlatformUtils::Initialize();

  auto parser = std::make_unique<XercesDOMParser>();
  parser->setValidationScheme( XercesDOMParser::Val_Always );
  parser->setDoNamespaces( true );
  parser->parse( fileName.fullPath().c_str() );

  if ( !parser )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "Cannot parse file `" << fileName.fullPath() << "' (parser = NULL).";
  
  DOMDocument* xmlDoc = parser->getDocument();

  if ( !xmlDoc )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "Cannot parse file `" << fileName.fullPath() << "' (xmlDoc = NULL).";

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  if ( !elementRoot )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "File `" << fileName.fullPath() << "' is empty.";

  RPAlignmentCorrectionsData corr_data = getCorrectionsData( elementRoot );

  XMLPlatformUtils::Terminate();

  return corr_data;
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionsData
RPAlignmentCorrectionsMethods::getCorrectionsData( DOMNode* root )
{
  RPAlignmentCorrectionsData result;

  DOMNodeList *children = root->getChildNodes();
  for ( unsigned int i = 0; i < children->getLength(); i++ ) {
    DOMNode *node = children->item( i );
    if ( node->getNodeType() != DOMNode::ELEMENT_NODE ) continue;
    const std::string node_name = cms::xerces::toString( node->getNodeName() );

    // check node type
    unsigned char nodeType = 0;
    if      ( node_name == "det" ) nodeType = 1;
    else if ( node_name == "rp"  ) nodeType = 2;

    if ( nodeType == 0 )
      throw cms::Exception("RPAlignmentCorrectionsMethods") << "Unknown node `" << cms::xerces::toString( node->getNodeName() ) << "'.";

    // check children
    if ( node->getChildNodes()->getLength() > 0 ) {
        edm::LogProblem("RPAlignmentCorrectionsMethods") << "LoadXMLFile > Warning: tag `" <<
          cms::xerces::toString( node->getNodeName() ) << "' has " << node->getChildNodes()->getLength() << " children nodes - they will be all ignored.";
    }

    // default values
    double sh_x = 0., sh_y = 0., sh_z = 0., rot_x = 0., rot_y = 0., rot_z = 0.;
    double sh_x_e = 0., sh_y_e = 0., sh_z_e = 0., rot_x_e = 0., rot_y_e = 0., rot_z_e = 0.;
    unsigned int id = 0;
    bool idSet = false;

    // get attributes
    DOMNamedNodeMap* attr = node->getAttributes();
    for ( unsigned int j = 0; j < attr->getLength(); j++ ) {
      DOMNode *a = attr->item( j );
      const std::string node_name = cms::xerces::toString( a->getNodeName() );

      if ( node_name == "id" ) {
        id = cms::xerces::toUInt( a->getNodeValue() );
        idSet = true;
      }
      else if ( node_name == "sh_x"    ) sh_x    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_x_e"  ) sh_x_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_y"    ) sh_y    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_y_e"  ) sh_y_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_z"    ) sh_z    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_z_e"  ) sh_z_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_x"   ) rot_x   = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_x_e" ) rot_x_e = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_y"   ) rot_y   = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_y_e" ) rot_y_e = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_z"   ) rot_z   = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_z_e" ) rot_z_e = cms::xerces::toDouble( a->getNodeValue() );
      else
        edm::LogProblem("RPAlignmentCorrectionsMethods") << ">> RPAlignmentCorrectionsMethods::LoadXMLFile > Warning: unknown attribute `"
          << cms::xerces::toString( a->getNodeName() ) << "'.";
    }

    // id must be set
    if ( !idSet )
        throw cms::Exception("RPAlignmentCorrectionsMethods") << "Id not set for tag `" << cms::xerces::toString( node->getNodeName() ) << "'.";

    // build alignment
    const RPAlignmentCorrectionData align_corr(
      sh_x*1e-3, sh_x_e*1e-3,
      sh_y*1e-3, sh_y_e*1e-3,
      sh_z*1e-3, sh_z_e*1e-3,
      rot_x*1e-3, rot_x_e*1e-3,
      rot_y*1e-3, rot_y_e*1e-3,
      rot_z*1e-3, rot_z_e*1e-3
    );

    // add the alignment to the right list
    if ( nodeType == 1 ) result.addSensorCorrection( id, align_corr, true );
    if ( nodeType == 2 ) result.addRPCorrection( id, align_corr, true );
  }

  return result;
}

//----------------------------------------------------------------------------------------------------

#define WRITE(q, dig, lim) \
  if (precise) \
    fprintf(f, " " #q "=\"%.15E\"", q()*1E3);\
  else \
    if (fabs(q()*1E3) < lim && q() != 0) \
      fprintf(f, " " #q "=\"%+8.1E\"", q()*1E3);\
    else \
      fprintf(f, " " #q "=\"%+8." #dig "f\"", q()*1E3);

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXML( const RPAlignmentCorrectionData& data, FILE* f, bool precise, bool wrErrors,
  bool wrSh_xy, bool wrSh_z, bool wrRot_xy, bool wrRot_z )
{
  if ( wrSh_xy )
  {
    WRITE( data.getShX, 2, 0.1 );
    WRITE( data.getShY, 2, 0.1 );
    if ( wrErrors )
    {
      WRITE( data.getShXUnc, 2, 0.1 );
      WRITE( data.getShYUnc, 2, 0.1 );
    }
  }

  if ( wrSh_z )
  {
    WRITE( data.getShZ, 2, 0.1 );
    if ( wrErrors )
    {
      WRITE( data.getShZUnc, 2, 0.1 );
    }
  }

  if ( wrRot_xy )
  {
    WRITE( data.getRotX, 3, 0.01 );
    WRITE( data.getRotY, 3, 0.01 );
    if ( wrErrors )
    {
      WRITE( data.getRotXUnc, 3, 0.01 );
      WRITE( data.getRotYUnc, 3, 0.01 );
    }
  }

  if ( wrRot_z )
  {
    WRITE( data.getRotZ, 3, 0.01 );
    if ( wrErrors )
    {
      WRITE( data.getRotZUnc, 3, 0.01 );
    }
  }
}

//----------------------------------------------------------------------------------------------------

#undef WRITE

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXMLFile( const RPAlignmentCorrectionsData& data, const std::string& fileName, bool precise, bool wrErrors,
  bool wrSh_xy, bool wrSh_z, bool wrRot_xy, bool wrRot_z )
{
  FILE* rf = fopen( fileName.c_str(), "w" );
  if ( !rf )
    throw cms::Exception("RPAlignmentCorrections::writeXMLFile") << "Cannot open file `" << fileName << "' to save alignments.";

  fprintf( rf, "<!-- Shifts in um, rotations in mrad. -->\n" );
  fprintf( rf, "<xml DocumentType=\"AlignmentDescription\">\n" );

  writeXMLBlock( data, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z );

  fprintf( rf, "</xml>\n" );
  fclose( rf );
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXMLBlock( const RPAlignmentCorrectionsData& data, FILE* rf, bool precise, bool wrErrors,
  bool wrSh_xy, bool wrSh_z, bool wrRot_xy, bool wrRot_z )
{
  bool firstRP = true;
  unsigned int prevRP = 0;
  std::set<unsigned int> writtenRPs;

  const auto &sensors = data.getSensorMap();
  const auto &rps = data.getRPMap();

  for (auto it = sensors.begin(); it != sensors.end(); ++it)
  {
    CTPPSDetId sensorId(it->first);
    unsigned int rpId = sensorId.getRPId();
    unsigned int decRPId = sensorId.arm()*100 + sensorId.station()*10 + sensorId.rp();

    // start a RP block
    if (firstRP || prevRP != rpId)
    {
      if (!firstRP)
        fprintf(rf, "\n");
      firstRP = false;

      fprintf(rf, "\t<!-- RP %3u -->\n", decRPId);

      auto rit = rps.find(rpId);
      if (rit != rps.end())
      {
        fprintf(rf, "\t<rp id=\"%u\" ", rit->first);
        writeXML( rit->second , rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z );
        fprintf(rf, "/>\n");
        writtenRPs.insert(rpId);
      }
    }
    prevRP = rpId;

    // write plane id
    unsigned int planeIdx = 1000;
    if (sensorId.subdetId() == CTPPSDetId::sdTrackingStrip) planeIdx = TotemRPDetId(it->first).plane();
    if (sensorId.subdetId() == CTPPSDetId::sdTrackingPixel) planeIdx = CTPPSPixelDetId(it->first).plane();
    if (sensorId.subdetId() == CTPPSDetId::sdTimingDiamond) planeIdx = CTPPSDiamondDetId(it->first).plane();
    fprintf(rf, "\t<!-- plane %u --> ", planeIdx);

    // write the correction
    fprintf(rf, "<det id=\"%u\"", it->first);
    writeXML(it->second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
    fprintf(rf, "/>\n");
  }

  // write remaining RPs
  for (auto it = rps.begin(); it != rps.end(); ++it)
  {
    std::set<unsigned int>::iterator wit = writtenRPs.find(it->first);
    if (wit == writtenRPs.end())
    {
      fprintf(rf, "\t<rp id=\"%u\"                                ", it->first);
      writeXML(it->second, rf, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
      fprintf(rf, "/>\n");
    }
  }
}
