/****************************************************************************
 *
 * This is a part of CMS-TOTEM PPS offline software.
 * Authors:
 *  Jan Kaspar (jan.kaspar@gmail.com)
 *  Helena Malbouisson
 *  Clemencia Mora Herrera
 *  Christopher Misan
 ****************************************************************************/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"
#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"  // this used to be RPMeasuredAlignmentRecord.h
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include <vector>
#include <string>
#include <map>
#include <set>

class CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon{
public:
    CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon(const edm::ParameterSet &p);
    ~CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon();
    CTPPSRPAlignmentCorrectionsDataSequence acsMeasured, acsReal, acsMisaligned;
    CTPPSRPAlignmentCorrectionsData acMeasured, acReal, acMisaligned;
    static edm::EventID previousLS(const edm::EventID &src);
    static edm::EventID nextLS(const edm::EventID &src);
    unsigned int verbosity;
protected:
    CTPPSRPAlignmentCorrectionsDataSequence Merge(const std::vector<CTPPSRPAlignmentCorrectionsDataSequence> &) const;
    void PrepareSequence(const std::string &label, CTPPSRPAlignmentCorrectionsDataSequence &seq, const std::vector<std::string> &files) const;

};