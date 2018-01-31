/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_RPAlignmentCorrectionsMethods
#define Geometry_VeryForwardGeometryBuilder_RPAlignmentCorrectionsMethods

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

#include <xercesc/dom/DOM.hpp>

class RPAlignmentCorrectionsMethods
{
  public:
    RPAlignmentCorrectionsMethods() {}

    static RPAlignmentCorrectionsData getCorrectionsDataFromFile( const edm::FileInPath& fileName );

    static RPAlignmentCorrectionsData getCorrectionsData( xercesc::DOMNode* );

    static void writeXML( const RPAlignmentCorrectionData& data, FILE* f, bool precise, bool wrErrors,
      bool wrSh_xy, bool wrSh_z, bool wrRot_xy, bool wrRot_z );

    /// writes corrections into a single XML file
    static void writeXMLFile( const RPAlignmentCorrectionsData&, const std::string& fileName, bool precise = false, bool wrErrors = true,
      bool wrSh_xy=true, bool wrSh_z=false, bool wrRot_xy=false, bool wrRot_z=true );

    /// writes a block of corrections into a file
    static void writeXMLBlock( const RPAlignmentCorrectionsData&, FILE*, bool precise = false, bool wrErrors = true,
      bool wrSh_xy=true, bool wrSh_z=false, bool wrRot_xy=false, bool wrRot_z=true );
};

#endif

