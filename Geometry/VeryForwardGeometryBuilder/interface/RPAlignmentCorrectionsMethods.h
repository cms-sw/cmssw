/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_RPAlignmentCorrectionsMethods
#define Geometry_VeryForwardGeometryBuilder_RPAlignmentCorrectionsMethods

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsDataSequence.h"

#include <xercesc/dom/DOM.hpp>

//----------------------------------------------------------------------------------------------------

class RPAlignmentCorrectionsMethods
{
  public:
    RPAlignmentCorrectionsMethods() {}

    /// loads sequence of alignment corrections from XML file
    static RPAlignmentCorrectionsDataSequence loadFromXML( const std::string& fileName );

    /// writes sequence of alignment corrections into a single XML file
    static void writeToXML( const RPAlignmentCorrectionsDataSequence &seq, const std::string& fileName,
      bool precise = false, bool wrErrors = true,
      bool wrSh_xy=true, bool wrSh_z=false, bool wrRot_xy=false, bool wrRot_z=true );

    /// writes alignment corrections into a single XML file, assigning infinite interval of validity
    static void writeToXML( const RPAlignmentCorrectionsData &ad, const std::string& fileName,
      bool precise = false, bool wrErrors = true,
      bool wrSh_xy=true, bool wrSh_z=false, bool wrRot_xy=false, bool wrRot_z=true )
    {
      const edm::ValidityInterval iov(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
      RPAlignmentCorrectionsDataSequence s;
      s.insert(iov, ad);
      writeToXML(s, fileName, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
    }

    static edm::IOVSyncValue stringToIOVValue(const std::string &);

    static std::string iovValueToString(const edm::IOVSyncValue &);

  protected:
    /// load corrections data corresponding to one IOV
    static RPAlignmentCorrectionsData getCorrectionsData( xercesc::DOMNode* );

    /// writes data of a correction in XML format
    static void writeXML( const RPAlignmentCorrectionData& data, FILE* f, bool precise, bool wrErrors,
      bool wrSh_xy, bool wrSh_z, bool wrRot_xy, bool wrRot_z );

    /// writes a block of corrections into a file
    static void writeXMLBlock( const RPAlignmentCorrectionsData&, FILE*, bool precise = false, bool wrErrors = true,
      bool wrSh_xy=true, bool wrSh_z=false, bool wrRot_xy=false, bool wrRot_z=true );
};

#endif
