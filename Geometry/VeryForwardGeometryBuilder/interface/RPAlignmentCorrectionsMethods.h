/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrections
#define Alignment_RPDataFormats_RPAlignmentCorrections

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

#include "Utilities/Xerces/interface/XercesStrUtils.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <map>
#include <set>

#include "TMatrixD.h"
#include "TVectorD.h"

//class AlignmentGeometry;

class RPAlignmentCorrectionsMethods
{
  public:
    RPAlignmentCorrectionsMethods() {}

    static RPAlignmentCorrectionsData getCorrectionsDataFromFile( const edm::FileInPath& fileName );
    static RPAlignmentCorrectionsData getCorrectionsData( xercesc::DOMNode* );

    static void writeXML( const RPAlignmentCorrectionData& data, FILE* f, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z );

    /// writes corrections into a single XML file
    static void writeXMLFile( const RPAlignmentCorrectionsData&, const std::string& fileName, bool precise = false, bool wrErrors = true, bool wrSh_r = true, bool wrSh_xy = true, bool wrSh_z = true, bool wrRot_z = true );

    /// writes a block of corrections into a file
    static void writeXMLBlock( const RPAlignmentCorrectionsData&, FILE*, bool precise = false, bool wrErrors = true, bool wrSh_r = true, bool wrSh_xy = true, bool wrSh_z = true, bool wrRot_z = true );

//    /// factors out the common shifts and rotations for every RP and saves these values as RPalignment
//    /// (factored variable), the expanded alignments are created as a by-product
//    static void FactorRPFromSensorCorrections(RPAlignmentCorrectionsData & data, RPAlignmentCorrectionsData &expanded, RPAlignmentCorrectionsData &factored,
//      const AlignmentGeometry &, bool equalWeights=false, unsigned int verbosity = 0);

};

#endif

