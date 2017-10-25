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
#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsMethods.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

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

void
RPAlignmentCorrectionsDataSequence::loadXMLFile( const std::string& fileName )
{
  // prepend CMSSW src dir
  const char* cmsswPath = getenv( "CMSSW_BASE" );
  size_t start = fileName.find_first_not_of( "   " );
  std::string fn = fileName.substr( start );
  if ( cmsswPath && fn[0] != '/' && fn.find( "./" ) != 0 )
    fn = std::string( cmsswPath ) + std::string( "/src/" ) + fn;

  // load DOM tree first the file
  try {
    XMLPlatformUtils::Initialize();
  } catch ( const XMLException& toCatch ) {
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "An XMLException caught with message: " << cms::xerces::toString( toCatch.getMessage() ) << ".";
  }

  auto parser = std::make_unique<XercesDOMParser>();
  parser->setValidationScheme( XercesDOMParser::Val_Always );
  parser->setDoNamespaces( true );
  parser->parse( fn.c_str() );

  if ( !parser )
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Cannot parse file `" << fn << "' (parser = NULL).";

  DOMDocument* xmlDoc = parser->getDocument();

  if ( !xmlDoc )
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Cannot parse file `" << fn << "' (xmlDoc = NULL).";

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  if ( !elementRoot )
    throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "File `" << fn << "' is empty.";

  // extract useful information form the DOM tree
  DOMNodeList* children = elementRoot->getChildNodes();
  for ( unsigned int i = 0; i < children->getLength(); i++ ) {
    DOMNode* node = children->item( i );
    if ( node->getNodeType() != DOMNode::ELEMENT_NODE ) continue;
    const std::string node_name = cms::xerces::toString( node->getNodeName() );

    // check node type
    unsigned char nodeType = 0;
    if      ( node_name == "TimeInterval" ) nodeType = 1;
    else if ( node_name == "det"          ) nodeType = 2;
    else if ( node_name == "rp"           ) nodeType = 3;

    if ( nodeType==0 )
      throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "Unknown node `" << cms::xerces::toString( node->getNodeName() ) << "'.";

    // old style - no TimeInterval block?
    if ( nodeType == 2 || nodeType == 3 ) {
      //edm::LogWarning("RPAlignmentCorrectionsDataSequence::LoadXMLFile") << "In file `" << fn << "' no TimeInterval given, assuming one block of infinite validity.";
      TimeValidityInterval inf;
      inf.SetInfinite();
      std::map<TimeValidityInterval,RPAlignmentCorrectionsData>::insert( std::make_pair( inf, RPAlignmentCorrectionsMethods::getCorrectionsData( elementRoot ) ) );
      break;
    }

    // get attributes
    edm::TimeValue_t first = 0, last = 0;
    bool first_set = false, last_set = false;
    DOMNamedNodeMap* attrs = node->getAttributes();
    for ( unsigned int j = 0; j < attrs->getLength(); j++ ) {
      const DOMNode* attr = attrs->item( j );
      const std::string node_name = cms::xerces::toString( node->getNodeName() );

      if ( node_name == "first" ) {
        first_set = true;
        first = TimeValidityInterval::UNIXStringToValue( cms::xerces::toString( attr->getNodeValue() ) );
      }
      else if ( node_name == "last" ) {
        last_set = true;
        last = TimeValidityInterval::UNIXStringToValue( cms::xerces::toString( attr->getNodeValue() ) );
      }
      else
        edm::LogProblem("RPAlignmentCorrectionsDataSequence") << ">> RPAlignmentCorrectionsDataSequence::LoadXMLFile > Warning: unknown attribute `"
          << cms::xerces::toString( attr->getNodeName() ) << "'.";
    }

    // interval of validity must be set
    if ( !first_set || !last_set )
      throw cms::Exception("RPAlignmentCorrectionsDataSequence") << "TimeInterval tag must have `first' and `last' attributes set.";

    TimeValidityInterval tvi( first, last );

    // process data
    RPAlignmentCorrectionsData corrections = RPAlignmentCorrectionsMethods::getCorrectionsData( node );

    // save result
    std::map<TimeValidityInterval,RPAlignmentCorrectionsData>::insert( std::make_pair( tvi, corrections ) );
  }

  parser.reset();
  XMLPlatformUtils::Terminate();
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsDataSequence::writeXMLFile( const std::string &fileName, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z ) const
{
  FILE *rf = fopen( fileName.c_str(), "w" );
  if ( !rf )
    throw cms::Exception("RPAlignmentCorrectionsDataSequence::writeXMLFile") << "Cannot open file `" << fileName
      << "' to save alignments.";

  fprintf( rf, "<!--\nShifts in um, rotations in mrad.\n\nFor more details see RPAlignmentCorrections::LoadXMLFile in\n" );
  fprintf( rf, "Alignment/RPDataFormats/src/RPAlignmentCorrectionsDataSequence.cc\n-->\n\n" );
  fprintf( rf, "<xml DocumentType=\"AlignmentSequenceDescription\">\n" );

  // write all time blocks
  for ( const auto& it : *this ) {
    fprintf( rf, "\t<TimeInterval first=\"%s\" last=\"%s\">",
      TimeValidityInterval::ValueToUNIXString( it.first.first ).c_str(),
      TimeValidityInterval::ValueToUNIXString( it.first.last ).c_str()
    );

    RPAlignmentCorrectionsMethods::writeXMLBlock( it.second, rf, precise, wrErrors, wrSh_r, wrSh_xy, wrSh_z, wrRot_z );
    fprintf( rf, "\t</TimeInterval>\n" );
  }

  fprintf( rf, "</xml>\n" );
  fclose( rf );
}

