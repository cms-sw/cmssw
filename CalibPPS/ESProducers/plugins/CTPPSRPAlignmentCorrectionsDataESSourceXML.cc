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
  std::unique_ptr<CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon> ctppsRPAlignmentCorrectionsDataESSourceXMLCommon;

  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceMeasured(const CTPPSRPAlignmentCorrectionsDataRcd &);
  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceReal(const RPRealAlignmentRecord &);
  std::unique_ptr<CTPPSRPAlignmentCorrectionsData> produceMisaligned(const RPMisalignedAlignmentRecord &);

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
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>(
      ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acMeasured);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSRPAlignmentCorrectionsData> CTPPSRPAlignmentCorrectionsDataESSourceXML::produceReal(
    const RPRealAlignmentRecord &iRecord) {
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acReal);
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSRPAlignmentCorrectionsData> CTPPSRPAlignmentCorrectionsDataESSourceXML::produceMisaligned(
    const RPMisalignedAlignmentRecord &iRecord) {
  return std::make_unique<CTPPSRPAlignmentCorrectionsData>(
      ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acMisaligned);
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
  CTPPSRPAlignmentCorrectionsDataSequence *p_seq = nullptr;
  CTPPSRPAlignmentCorrectionsData *p_corr = nullptr;

  if (strcmp(key.name(), "CTPPSRPAlignmentCorrectionsDataRcd") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMeasured);
    p_corr = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acMeasured);
  }

  if (strcmp(key.name(), "RPRealAlignmentRecord") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsReal);
    p_corr = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acReal);
  }

  if (strcmp(key.name(), "RPMisalignedAlignmentRecord") == 0) {
    p_seq = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acsMisaligned);
    p_corr = &(ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->acMisaligned);
  }

  if (p_corr == nullptr)
    throw cms::Exception("CTPPSRPAlignmentCorrectionsDataESSourceXML::setIntervalFor")
        << "Unknown record " << key.name();

  // // find the corresponding interval
  bool next_exists = false;
  const edm::EventID &event_curr = iosv.eventID();
  edm::EventID event_next_start(edm::EventID::maxRunNumber(), edm::EventID::maxLuminosityBlockNumber(), 1);

  for (const auto &it : *p_seq) {
    const auto &it_event_first = it.first.first().eventID();
    const auto &it_event_last = it.first.last().eventID();

    bool it_contained_lo = ((it_event_first.run() < event_curr.run()) ||
                            ((it_event_first.run() == event_curr.run()) &&
                             (it_event_first.luminosityBlock() <= event_curr.luminosityBlock())));

    bool it_contained_up = ((it_event_last.run() > event_curr.run()) ||
                            ((it_event_last.run() == event_curr.run()) &&
                             (it_event_last.luminosityBlock() >= event_curr.luminosityBlock())));

    if (it_contained_lo && it_contained_up) {
      valInt = it.first;
      *p_corr = it.second;

      if (ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->verbosity) {
        LogInfo("PPS") << "    setting validity interval ["
                       << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.first()) << ", "
                       << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.last()) << "]";
      }

      return;
    }

    bool it_in_future = ((it_event_first.run() > event_curr.run()) ||
                         ((it_event_first.run() == event_curr.run() &&
                           (it_event_first.luminosityBlock() > event_curr.luminosityBlock()))));

    if (it_in_future) {
      next_exists = true;
      if (event_next_start > it_event_first)
        event_next_start = it_event_first;
    }
  }

  // no interval found, set empty corrections
  *p_corr = CTPPSRPAlignmentCorrectionsData();

  if (!next_exists) {
    valInt = ValidityInterval(iosv, iosv.endOfTime());
  } else {
    const EventID &event_last = ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->previousLS(event_next_start);
    valInt = ValidityInterval(iosv, IOVSyncValue(event_last));
  }

  if (ctppsRPAlignmentCorrectionsDataESSourceXMLCommon->verbosity) {
    LogInfo("PPS") << "    setting validity interval ["
                   << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.first()) << ", "
                   << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.last())
                   << "] (empty alignment corrections)";
  }
}

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSRPAlignmentCorrectionsDataESSourceXML);
