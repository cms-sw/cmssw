/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Seyed Mohsen Etesami (setesami@cern.ch)
****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/CTPPSRawToDigi/interface/RawToDigiConverter.h"
#include "EventFilter/CTPPSRawToDigi/interface/CounterChecker.h"
#include "EventFilter/CTPPSRawToDigi/interface/DiamondVFATFrame.h"
#include "EventFilter/CTPPSRawToDigi/interface/TotemSampicFrame.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"

using namespace std;
using namespace edm;

RawToDigiConverter::RawToDigiConverter(const edm::ParameterSet &conf)
    : verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
      printErrorSummary(conf.getUntrackedParameter<unsigned int>("printErrorSummary", 1)),
      printUnknownFrameSummary(conf.getUntrackedParameter<unsigned int>("printUnknownFrameSummary", 1)),

      testFootprint(conf.getParameter<unsigned int>("testFootprint")),
      testCRC(conf.getParameter<unsigned int>("testCRC")),
      testID(conf.getParameter<unsigned int>("testID")),
      testECMostFrequent(conf.getParameter<unsigned int>("testECMostFrequent")),
      testBCMostFrequent(conf.getParameter<unsigned int>("testBCMostFrequent")),

      EC_min(conf.getUntrackedParameter<unsigned int>("EC_min", 10)),
      BC_min(conf.getUntrackedParameter<unsigned int>("BC_min", 10)),

      EC_fraction(conf.getUntrackedParameter<double>("EC_fraction", 0.6)),
      BC_fraction(conf.getUntrackedParameter<double>("BC_fraction", 0.6)) {}

void RawToDigiConverter::runCommon(const VFATFrameCollection &input,
                                   const TotemDAQMapping &mapping,
                                   map<TotemFramePosition, RawToDigiConverter::Record> &records) {
  // EC and BC checks (wrt. the most frequent value), BC checks per subsystem
  CounterChecker ECChecker(CounterChecker::ECChecker, "EC", EC_min, EC_fraction, verbosity);
  CounterChecker BCChecker(CounterChecker::BCChecker, "BC", BC_min, BC_fraction, verbosity);

  // initialise structure merging vfat frame data with the mapping
  for (auto &p : mapping.VFATMapping) {
    TotemVFATStatus st;
    st.setMissing(true);
    records[p.first] = {&p.second, nullptr, st};
  }

  // event error message buffer
  stringstream ees;

  // associate data frames with records
  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next()) {
    // frame error message buffer
    stringstream fes;

    bool problemsPresent = false;
    bool stopProcessing = false;

    // skip data frames not listed in the DAQ mapping
    auto records_it = records.find(fr.Position());
    if (records_it == records.end()) {
      unknownSummary[fr.Position()]++;
      continue;
    }

    // update record
    Record &record = records_it->second;
    record.frame = fr.Data();
    record.status.setMissing(false);
    record.status.setNumberOfClustersSpecified(record.frame->isNumberOfClustersPresent());
    record.status.setNumberOfClusters(record.frame->getNumberOfClusters());

    // check footprint
    if (testFootprint != tfNoTest && !record.frame->checkFootprint()) {
      problemsPresent = true;

      if (verbosity > 0)
        fes << "    invalid footprint" << endl;

      if (testFootprint == tfErr) {
        record.status.setFootprintError();
        stopProcessing = true;
      }
    }

    // check CRC
    if (testCRC != tfNoTest && !record.frame->checkCRC()) {
      problemsPresent = true;

      if (verbosity > 0)
        fes << "    CRC failure" << endl;

      if (testCRC == tfErr) {
        record.status.setCRCError();
        stopProcessing = true;
      }
    }
    // check the id mismatch
    if (testID != tfNoTest && record.frame->isIDPresent() &&
        (record.frame->getChipID() & 0xFFF) != (record.info->hwID & 0xFFF)) {
      if (verbosity > 0)
        fes << "    ID mismatch (data: 0x" << hex << record.frame->getChipID() << ", mapping: 0x" << record.info->hwID
            << dec << ", symbId: " << record.info->symbolicID.symbolicID << ")" << endl;

      if (testID == tfErr) {
        record.status.setIDMismatch();
        stopProcessing = true;
      }
    }

    // if there were errors, put the information to ees buffer
    if (verbosity > 0 && problemsPresent) {
      string message = (stopProcessing) ? "(and will be dropped)" : "(but will be used though)";
      if (verbosity > 2) {
        ees << "  Frame at " << fr.Position() << " seems corrupted " << message << ":" << endl;
        ees << fes.rdbuf();
      } else
        ees << "  Frame at " << fr.Position() << " seems corrupted " << message << "." << endl;
    }

    // if there were serious errors, do not process this frame
    if (stopProcessing)
      continue;

    // fill EC and BC values to the statistics
    if (fr.Data()->isECPresent())
      ECChecker.Fill(fr.Data()->getEC(), fr.Position());

    if (fr.Data()->isBCPresent())
      BCChecker.Fill(fr.Data()->getBC(), fr.Position());
  }

  // analyze EC and BC statistics
  if (testECMostFrequent != tfNoTest)
    ECChecker.Analyze(records, (testECMostFrequent == tfErr), ees);

  if (testBCMostFrequent != tfNoTest)
    BCChecker.Analyze(records, (testBCMostFrequent == tfErr), ees);

  // add error message for missing frames
  if (verbosity > 1) {
    for (const auto &p : records) {
      if (p.second.status.isMissing())
        ees << "Frame for VFAT " << p.first << " is not present in the data." << endl;
    }
  }

  // print error message
  if (verbosity > 0 && !ees.rdbuf()->str().empty()) {
    if (verbosity > 1)
      LogWarning("Totem") << "Error in RawToDigiConverter::runCommon > "
                          << "event contains the following problems:" << endl
                          << ees.rdbuf() << endl;
    else
      LogWarning("Totem") << "Error in RawToDigiConverter::runCommon > "
                          << "event contains problems." << endl;
  }

  // increase error counters
  if (printErrorSummary) {
    for (const auto &it : records) {
      if (!it.second.status.isOK()) {
        auto &m = errorSummary[it.first];
        m[it.second.status]++;
      }
    }
  }
}

void RawToDigiConverter::run(const VFATFrameCollection &input,
                             const TotemDAQMapping &mapping,
                             const TotemAnalysisMask &analysisMask,
                             DetSetVector<TotemRPDigi> &rpData,
                             DetSetVector<TotemVFATStatus> &finalStatus) {
  // structure merging vfat frame data with the mapping
  map<TotemFramePosition, Record> records;

  // common processing - frame validation
  runCommon(input, mapping, records);

  // second loop over data
  for (auto &p : records) {
    Record &record = p.second;

    // calculate ids
    TotemRPDetId chipId(record.info->symbolicID.symbolicID);
    uint8_t chipPosition = chipId.chip();
    TotemRPDetId detId = chipId.planeId();

    // update chipPosition in status
    record.status.setChipPosition(chipPosition);

    // produce digi only for good frames
    if (record.status.isOK()) {
      // find analysis mask (needs a default=no mask, if not in present the mapping)
      TotemVFATAnalysisMask anMa;
      anMa.fullMask = false;

      auto analysisIter = analysisMask.analysisMask.find(record.info->symbolicID);
      if (analysisIter != analysisMask.analysisMask.end()) {
        // if there is some information about masked channels - save it into conversionStatus
        anMa = analysisIter->second;
        if (anMa.fullMask)
          record.status.setFullyMaskedOut();
        else
          record.status.setPartiallyMaskedOut();
      }

      // create the digi
      unsigned short offset = chipPosition * 128;
      const vector<unsigned char> &activeChannels = record.frame->getActiveChannels();

      for (auto ch : activeChannels) {
        // skip masked channels
        if (!anMa.fullMask && anMa.maskedChannels.find(ch) == anMa.maskedChannels.end()) {
          DetSet<TotemRPDigi> &digiDetSet = rpData.find_or_insert(detId);
          digiDetSet.push_back(TotemRPDigi(offset + ch));
        }
      }
    }

    // save status
    DetSet<TotemVFATStatus> &statusDetSet = finalStatus.find_or_insert(detId);
    statusDetSet.push_back(record.status);
  }
}

void RawToDigiConverter::run(const VFATFrameCollection &coll,
                             const TotemDAQMapping &mapping,
                             const TotemAnalysisMask &mask,
                             edm::DetSetVector<CTPPSDiamondDigi> &digi,
                             edm::DetSetVector<TotemVFATStatus> &status) {
  // structure merging vfat frame data with the mapping
  map<TotemFramePosition, Record> records;

  // common processing - frame validation
  runCommon(coll, mapping, records);

  // second loop over data
  for (auto &p : records) {
    Record &record = p.second;

    // calculate ids
    CTPPSDiamondDetId detId(record.info->symbolicID.symbolicID);

    if (record.status.isOK()) {
      // update Event Counter in status
      record.status.setEC(record.frame->getEC() & 0xFF);

      // create the digi
      DetSet<CTPPSDiamondDigi> &digiDetSet = digi.find_or_insert(detId);
      digiDetSet.emplace_back(pps::diamond::vfat::getLeadingEdgeTime(*record.frame),
                              pps::diamond::vfat::getTrailingEdgeTime(*record.frame),
                              pps::diamond::vfat::getThresholdVoltage(*record.frame),
                              pps::diamond::vfat::getMultihit(*record.frame),
                              pps::diamond::vfat::getHptdcErrorFlag(*record.frame));
    }

    // save status
    DetSet<TotemVFATStatus> &statusDetSet = status.find_or_insert(detId);
    statusDetSet.push_back(record.status);
  }
}

void RawToDigiConverter::run(const VFATFrameCollection &coll,
                             const TotemDAQMapping &mapping,
                             const TotemAnalysisMask &mask,
                             edm::DetSetVector<TotemTimingDigi> &digi,
                             edm::DetSetVector<TotemVFATStatus> &status) {
  // structure merging vfat frame data with the mapping
  map<TotemFramePosition, Record> records;

  // common processing - frame validation
  runCommon(coll, mapping, records);

  // second loop over data
  for (auto &p : records) {
    Record &record = p.second;
    if (!record.status.isOK())
      continue;

    const TotemFramePosition *framepos = &p.first;

    if (((framepos->getIdxInFiber() % 2) == 0) && (framepos->getIdxInFiber() < 14)) {
      //corresponding channel data are always in the neighbouring idx in fiber

      TotemFramePosition frameposdata(framepos->getSubSystemId(),
                                      framepos->getTOTFEDId(),
                                      framepos->getOptoRxId(),
                                      framepos->getGOHId(),
                                      (framepos->getIdxInFiber() + 1));
      TotemFramePosition frameposEvtInfo(
          framepos->getSubSystemId(), framepos->getTOTFEDId(), framepos->getOptoRxId(), framepos->getGOHId(), 0xe);

      auto channelwaveformPtr = records.find(frameposdata);
      auto eventInfoPtr = records.find(frameposEvtInfo);

      if (channelwaveformPtr != records.end() && eventInfoPtr != records.end()) {
        Record &channelwaveform = records[frameposdata];
        Record &eventInfo = records[frameposEvtInfo];

        // Extract all the waveform information from the raw data
        TotemSampicFrame totemSampicFrame((const uint8_t *)record.frame->getData(),
                                          (const uint8_t *)channelwaveform.frame->getData(),
                                          (const uint8_t *)eventInfo.frame->getData());

        if (totemSampicFrame.valid()) {
          // create the digi
          TotemTimingEventInfo eventInfoTmp(totemSampicFrame.getEventHardwareId(),
                                            totemSampicFrame.getL1ATimestamp(),
                                            totemSampicFrame.getBunchNumber(),
                                            totemSampicFrame.getOrbitNumber(),
                                            totemSampicFrame.getEventNumber(),
                                            totemSampicFrame.getChannelMap(),
                                            totemSampicFrame.getL1ALatency(),
                                            totemSampicFrame.getNumberOfSentSamples(),
                                            totemSampicFrame.getOffsetOfSamples(),
                                            totemSampicFrame.getPLLInfo());
          TotemTimingDigi digiTmp(totemSampicFrame.getHardwareId(),
                                  totemSampicFrame.getFPGATimestamp(),
                                  totemSampicFrame.getTimestampA(),
                                  totemSampicFrame.getTimestampB(),
                                  totemSampicFrame.getCellInfo(),
                                  totemSampicFrame.getSamples(),
                                  eventInfoTmp);
          // calculate ids
          TotemTimingDetId detId(record.info->symbolicID.symbolicID);

          const TotemDAQMapping::TotemTimingPlaneChannelPair SWpair =
              mapping.getTimingChannel(totemSampicFrame.getHardwareId());
          // for FW Version > 0 plane and channel are encoded in the dataframe
          if (totemSampicFrame.getFWVersion() < 0x30)  // Mapping not present in HW, read from SW for FW versions < 3.0
          {
            if (SWpair.plane == -1 || SWpair.channel == -1) {
              if (verbosity > 0)
                LogWarning("Totem") << "Error in RawToDigiConverter::TotemTiming > "
                                    << "HwId not recognized!  hwId: " << std::hex
                                    << (unsigned int)totemSampicFrame.getHardwareId() << endl;
            } else {
              detId.setPlane(SWpair.plane % 4);
              detId.setChannel(SWpair.channel);
              detId.setRP(SWpair.plane / 4);  // Top:0 or Bottom:1
            }
          } else  // Mapping read from HW, checked by SW
          {
            const int HWplane = totemSampicFrame.getDetPlane() % 16;
            const int HWchannel = totemSampicFrame.getDetChannel() % 16;

            if (SWpair.plane == -1 || SWpair.channel == -1) {
              if (verbosity > 0)
                LogWarning("Totem") << "Warning in RawToDigiConverter::TotemTiming > "
                                    << "HwId not recognized!  hwId: " << std::hex
                                    << (unsigned int)totemSampicFrame.getHardwareId()
                                    << "\tUsing plane and ch from HW without check!" << endl;
            } else {
              if (verbosity > 0 && (SWpair.plane != HWplane || SWpair.channel != HWchannel))
                LogWarning("Totem") << "Warning in RawToDigiConverter::TotemTiming > "
                                    << "Hw mapping different from SW mapping. hwId: " << std::hex
                                    << (unsigned int)totemSampicFrame.getHardwareId() << "HW: " << std::dec << HWplane
                                    << ":" << HWchannel << "\tSW " << SWpair.plane << ":" << SWpair.channel
                                    << "\tUsing plane and ch from HW!" << endl;
            }
            detId.setPlane(HWplane % 4);
            detId.setChannel(HWchannel);
            detId.setRP(HWplane / 4);  // Top:0 or Bottom:1
          }

          DetSet<TotemTimingDigi> &digiDetSet = digi.find_or_insert(detId);
          digiDetSet.push_back(digiTmp);
        }
      }
    }
  }
}

void RawToDigiConverter::printSummaries() const {
  // print error summary
  if (printErrorSummary) {
    if (!errorSummary.empty()) {
      stringstream ees;
      for (const auto &vit : errorSummary) {
        ees << vit.first << endl;

        for (const auto &it : vit.second)
          ees << "    " << it.first << " : " << it.second << endl;
      }

      LogWarning("Totem") << "RawToDigiConverter: error summary (error signature : number of such events)\n"
                          << endl
                          << ees.rdbuf();
    } else {
      LogInfo("Totem") << "RawToDigiConverter: no errors to be reported.";
    }
  }

  // print summary of unknown frames (found in data but not in the mapping)
  if (printUnknownFrameSummary) {
    if (!unknownSummary.empty()) {
      stringstream ees;
      for (const auto &it : unknownSummary)
        ees << "  " << it.first << " : " << it.second << endl;

      LogWarning("Totem")
          << "RawToDigiConverter: frames found in data, but not in the mapping (frame position : number of events)\n"
          << endl
          << ees.rdbuf();
    } else {
      LogInfo("Totem") << "RawToDigiConverter: no unknown frames to be reported.";
    }
  }
}
