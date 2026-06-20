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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"  // this used to be RPMeasuredAlignmentRecord.h
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

#include "CalibPPS/ESProducers/interface/CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon.h"

#include <vector>
#include <string>
#include <map>
#include <set>

using namespace std;
using namespace edm;

/**
 * Loads alignment corrections to EventSetup.
 **/
class CTPPSRPAlignmentCorrectionsDataESSourceXML : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CTPPSRPAlignmentCorrectionsDataESSourceXML(const edm::ParameterSet &p);
  ~CTPPSRPAlignmentCorrectionsDataESSourceXML() override;

protected:
  std::unique_ptr<CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon const>
      ctppsRPAlignmentCorrectionsDataESSourceXMLCommon;

  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceMeasured(const CTPPSRPAlignmentCorrectionsDataRcd &);
  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceReal(const RPRealAlignmentRecord &);
  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceMisaligned(const RPMisalignedAlignmentRecord &);

  bool isConcurrentFinder() const override { return true; }
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
};

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataESSourceXML::CTPPSRPAlignmentCorrectionsDataESSourceXML(const edm::ParameterSet &pSet) {
  ctppsRPAlignmentCorrectionsDataESSourceXMLCommon =
      std::make_unique<CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon>(pSet);
  setWhatProduced(this, &CTPPSRPAlignmentCorrectionsDataESSourceXML::produceMeasured);
  setWhatProduced(this, &CTPPSRPAlignmentCorrectionsDataESSourceXML::produceReal);
  setWhatProduced(this, &CTPPSRPAlignmentCorrectionsDataESSourceXML::produceMisaligned);

  findingRecord<CTPPSRPAlignmentCorrectionsDataRcd>();
  findingRecord<RPRealAlignmentRecord>();
  findingRecord<RPMisalignedAlignmentRecord>();
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataESSourceXML::~CTPPSRPAlignmentCorrectionsDataESSourceXML() {}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSRPAlignmentCorrectionsData> CTPPSRPAlignmentCorrectionsDataESSourceXML::produceMeasured(
    const CTPPSRPAlignmentCorrectionsDataRcd &iRecord) {
  auto data = CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::dataFor(
      iRecord.validityInterval().first(), ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMeasured);
  if (data) {
    return std::make_unique<CTPPSRPAlignmentCorrectionsData>(*data);
  }
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>();
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSRPAlignmentCorrectionsData> CTPPSRPAlignmentCorrectionsDataESSourceXML::produceReal(
    const RPRealAlignmentRecord &iRecord) {
  auto data = CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::dataFor(
      iRecord.validityInterval().first(), ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsReal);
  if (data) {
    return std::make_unique<CTPPSRPAlignmentCorrectionsData>(*data);
  }
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>();
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSRPAlignmentCorrectionsData> CTPPSRPAlignmentCorrectionsDataESSourceXML::produceMisaligned(
    const RPMisalignedAlignmentRecord &iRecord) {
  auto data = CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::dataFor(
      iRecord.validityInterval().first(), ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMisaligned);
  if (data) {
    return std::make_unique<CTPPSRPAlignmentCorrectionsData>(*data);
  }
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsDataESSourceXML::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                                const IOVSyncValue &iosv,
                                                                ValidityInterval &valInt) {
  if (ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->verbosity) {
    time_t unixTime = iosv.time().unixTime();
    char timeStr[50];
    strftime(timeStr, 50, "%F %T", localtime(&unixTime));

    LogInfo("PPS") << ">> CTPPSRPAlignmentCorrectionsDataESSourceXML::setIntervalFor(" << key.name() << ")";

    LogInfo("PPS") << "    event=" << iosv.eventID() << ", UNIX timestamp=" << unixTime << " (" << timeStr << ")";
  }

  // // determine what sequence and corrections should be used
  CTPPSRPAlignmentCorrectionsDataSequence const *p_seq = nullptr;

  bool knownRecord = false;
  if (strcmp(key.name(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMeasured);
    knownRecord = true;
  }

  if (strcmp(key.name(), "RPRealAlignmentRecord") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsReal);
    knownRecord = true;
  }

  if (strcmp(key.name(), "RPMisalignedAlignmentRecord") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMisaligned);
    knownRecord = true;
  }

  if (not knownRecord)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsDataESSourceXML::setIntervalFor")
        << "Unknown record " << key.name();

  valInt = CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::intervalFor(
      iosv, *p_seq, ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->verbosity);
  if (ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->verbosity) {
    LogInfo("PPS") << "    setting validity interval ["
                   << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.first()) << ", "
                   << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.last())
                   << "] (empty alignment corrections)";
  }
}

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSRPAlignmentCorrectionsDataESSourceXML);
