/****************************************************************************
 * Authors:
 *  Jan Kaspar (jan.kaspar@gmail.com)
 *  Helena Malbouisson
 *  Clemencia Mora Herrera
 *  Christopher Misan
 ****************************************************************************/

#include "CalibPPS/ESProducers/interface/CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include <map>
#include <set>

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon(
    const edm::ParameterSet &pSet)
    : verbosity(pSet.getUntrackedParameter<unsigned int>("verbosity", 0)) {
  std::vector<std::string> measuredFiles;
  for (const auto &f : pSet.getParameter<std::vector<std::string> >("MeasuredFiles"))
    measuredFiles.push_back(edm::FileInPath(f).fullPath());
  PrepareSequence("Measured", acsMeasured, measuredFiles);

  std::vector<std::string> realFiles;
  for (const auto &f : pSet.getParameter<std::vector<std::string> >("RealFiles"))
    realFiles.push_back(edm::FileInPath(f).fullPath());
  PrepareSequence("Real", acsReal, realFiles);

  std::vector<std::string> misalignedFiles;
  for (const auto &f : pSet.getParameter<std::vector<std::string> >("MisalignedFiles"))
    misalignedFiles.push_back(edm::FileInPath(f).fullPath());
  PrepareSequence("Misaligned", acsMisaligned, misalignedFiles);
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::~CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon() {}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsDataSequence CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::Merge(
    const std::vector<CTPPSRPAlignmentCorrectionsDataSequence> &seqs) const {
  // find interval boundaries
  std::map<edm::EventID, std::vector<std::pair<bool, const CTPPSRPAlignmentCorrectionsData *> > > bounds;

  for (const auto &seq : seqs) {
    for (const auto &p : seq) {
      const edm::ValidityInterval &iov = p.first;
      const CTPPSRPAlignmentCorrectionsData *corr = &p.second;

      const edm::EventID &event_first = iov.first().eventID();
      bounds[event_first].emplace_back(std::pair<bool, const CTPPSRPAlignmentCorrectionsData *>(true, corr));

      const edm::EventID &event_after = nextLS(iov.last().eventID());
      bounds[event_after].emplace_back(std::pair<bool, const CTPPSRPAlignmentCorrectionsData *>(false, corr));
    }
  }

  // build correction sums per interval
  std::set<const CTPPSRPAlignmentCorrectionsData *> accumulator;
  CTPPSRPAlignmentCorrectionsDataSequence result;
  for (std::map<edm::EventID, std::vector<std::pair<bool, const CTPPSRPAlignmentCorrectionsData *> > >::const_iterator
           tit = bounds.begin();
       tit != bounds.end();
       ++tit) {
    for (const auto &cit : tit->second) {
      bool add = cit.first;
      const CTPPSRPAlignmentCorrectionsData *corr = cit.second;

      if (add)
        accumulator.insert(corr);
      else
        accumulator.erase(corr);
    }

    auto tit_next = tit;
    tit_next++;
    if (tit_next == bounds.end())
      break;

    const edm::EventID &event_first = tit->first;
    const edm::EventID &event_last = previousLS(tit_next->first);

    if (verbosity) {
      edm::LogInfo("PPS") << "    first="
                          << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(edm::IOVSyncValue(event_first))
                          << ", last="
                          << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(edm::IOVSyncValue(event_last))
                          << ": alignment blocks " << accumulator.size();
    }

    CTPPSRPAlignmentCorrectionsData corr_sum;
    for (auto sit : accumulator)
      corr_sum.addCorrections(*sit);

    result.insert(edm::ValidityInterval(edm::IOVSyncValue(event_first), edm::IOVSyncValue(event_last)), corr_sum);
  }

  return result;
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::PrepareSequence(const std::string &label,
                                                                       CTPPSRPAlignmentCorrectionsDataSequence &seq,
                                                                       const std::vector<std::string> &files) const {
  if (verbosity)
    edm::LogInfo("PPS") << "PrepareSequence(" << label << ")";

  std::vector<CTPPSRPAlignmentCorrectionsDataSequence> sequences;
  sequences.reserve(files.size());
  for (const auto &file : files)
    sequences.emplace_back(CTPPSRPAlignmentCorrectionsMethods::loadFromXML(file));

  seq = Merge(sequences);
}

//----------------------------------------------------------------------------------------------------

edm::EventID CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::previousLS(const edm::EventID &src) {
  if (src.run() == edm::EventID::maxRunNumber() && src.luminosityBlock() == edm::EventID::maxLuminosityBlockNumber())
    return src;

  if (src.luminosityBlock() == 0)
    return edm::EventID(src.run() - 1, edm::EventID::maxLuminosityBlockNumber(), src.event());

  return edm::EventID(src.run(), src.luminosityBlock() - 1, src.event());
}

//----------------------------------------------------------------------------------------------------

edm::EventID CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::nextLS(const edm::EventID &src) {
  if (src.luminosityBlock() == edm::EventID::maxLuminosityBlockNumber()) {
    if (src.run() == edm::EventID::maxRunNumber())
      return src;

    return edm::EventID(src.run() + 1, 0, src.event());
  }

  return edm::EventID(src.run(), src.luminosityBlock() + 1, src.event());
}

//----------------------------------------------------------------------------------------------------
const CTPPSRPAlignmentCorrectionsData *CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::dataFor(
    edm::IOVSyncValue const &iosv, CTPPSRPAlignmentCorrectionsDataSequence const &seq) {
  const edm::EventID &event_curr = iosv.eventID();

  for (const auto &it : seq) {
    const auto &it_event_first = it.first.first().eventID();
    const auto &it_event_last = it.first.last().eventID();

    bool it_contained_lo = ((it_event_first.run() < event_curr.run()) ||
                            ((it_event_first.run() == event_curr.run()) &&
                             (it_event_first.luminosityBlock() <= event_curr.luminosityBlock())));

    bool it_contained_up = ((it_event_last.run() > event_curr.run()) ||
                            ((it_event_last.run() == event_curr.run()) &&
                             (it_event_last.luminosityBlock() >= event_curr.luminosityBlock())));

    if (it_contained_lo && it_contained_up) {
      return &it.second;
    }
  }
  return nullptr;
}
//----------------------------------------------------------------------------------------------------
edm::ValidityInterval CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon::intervalFor(
    edm::IOVSyncValue const &iosv, CTPPSRPAlignmentCorrectionsDataSequence const &seq, bool verbosity) {
  // // find the corresponding interval
  bool next_exists = false;
  const edm::EventID &event_curr = iosv.eventID();
  edm::EventID event_next_start(edm::EventID::maxRunNumber(), edm::EventID::maxLuminosityBlockNumber(), 1);

  for (const auto &it : seq) {
    const auto &it_event_first = it.first.first().eventID();
    const auto &it_event_last = it.first.last().eventID();

    bool it_contained_lo = ((it_event_first.run() < event_curr.run()) ||
                            ((it_event_first.run() == event_curr.run()) &&
                             (it_event_first.luminosityBlock() <= event_curr.luminosityBlock())));

    bool it_contained_up = ((it_event_last.run() > event_curr.run()) ||
                            ((it_event_last.run() == event_curr.run()) &&
                             (it_event_last.luminosityBlock() >= event_curr.luminosityBlock())));

    if (it_contained_lo && it_contained_up) {
      auto valInt = it.first;

      if (verbosity) {
        edm::LogInfo("PPS") << "    setting validity interval ["
                            << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.first()) << ", "
                            << CTPPSRPAlignmentCorrectionsMethods::iovValueToString(valInt.last()) << "]";
      }
      return valInt;
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

  if (!next_exists) {
    return edm::ValidityInterval(iosv, iosv.endOfTime());
  }
  const edm::EventID &event_last = previousLS(event_next_start);
  return edm::ValidityInterval(iosv, edm::IOVSyncValue(event_last));
}
