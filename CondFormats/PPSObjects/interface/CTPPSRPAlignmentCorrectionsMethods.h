/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 * Jan Kaspar
 * Helena Malbouisson
 * Clemencia Mora Herrera
 *
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_CTPPSRPAlignmentCorrectionsMethods
#define CondFormats_PPSObjects_CTPPSRPAlignmentCorrectionsMethods

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"

#include <xercesc/dom/DOM.hpp>

//----------------------------------------------------------------------------------------------------

class CTPPSRPAlignmentCorrectionsMethods {
public:
  CTPPSRPAlignmentCorrectionsMethods() {}

  /// loads sequence of alignment corrections from XML file
  static CTPPSRPAlignmentCorrectionsDataSequence loadFromXML(const std::string& fileName);

  /// writes sequence of alignment corrections into a single XML file
  static void writeToXML(const CTPPSRPAlignmentCorrectionsDataSequence& seq,
                         const std::string& fileName,
                         bool precise = false,
                         bool wrErrors = true,
                         bool wrSh_xy = true,
                         bool wrSh_z = false,
                         bool wrRot_xy = false,
                         bool wrRot_z = true);

  /// writes alignment corrections into a single XML file, assigning infinite interval of validity
  static void writeToXML(const CTPPSRPAlignmentCorrectionsData& ad,
                         const std::string& fileName,
                         bool precise = false,
                         bool wrErrors = true,
                         bool wrSh_xy = true,
                         bool wrSh_z = false,
                         bool wrRot_xy = false,
                         bool wrRot_z = true) {
    const edm::ValidityInterval iov(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
    CTPPSRPAlignmentCorrectionsDataSequence s;
    s.insert(iov, ad);
    writeToXML(s, fileName, precise, wrErrors, wrSh_xy, wrSh_z, wrRot_xy, wrRot_z);
  }

  static edm::IOVSyncValue stringToIOVValue(const std::string&);

  static std::string iovValueToString(const edm::IOVSyncValue&);

protected:
  /// load corrections data corresponding to one IOV
  static CTPPSRPAlignmentCorrectionsData getCorrectionsData(xercesc::DOMNode*);

  /// writes data of a correction in XML format
  static void writeXML(const CTPPSRPAlignmentCorrectionData& data,
                       FILE* f,
                       bool precise,
                       bool wrErrors,
                       bool wrSh_xy,
                       bool wrSh_z,
                       bool wrRot_xy,
                       bool wrRot_z);

  /// writes a block of corrections into a file
  static void writeXMLBlock(const CTPPSRPAlignmentCorrectionsData&,
                            FILE*,
                            bool precise = false,
                            bool wrErrors = true,
                            bool wrSh_xy = true,
                            bool wrSh_z = false,
                            bool wrRot_xy = false,
                            bool wrRot_z = true);
};

#endif
