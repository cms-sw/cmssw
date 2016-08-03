/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/RawToDigiConverter.h"

#include "EventFilter/TotemRawToDigi/interface/CounterChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

RawToDigiConverter::RawToDigiConverter(const edm::ParameterSet &conf) :
  verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
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
  BC_fraction(conf.getUntrackedParameter<double>("BC_fraction", 0.6))
{
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::RunCommon(const VFATFrameCollection &input, const TotemDAQMapping &mapping,
      map<TotemFramePosition, RawToDigiConverter::Record> &records)
{
  // EC and BC checks (wrt. the most frequent value), BC checks per subsystem
  CounterChecker ECChecker(CounterChecker::ECChecker, "EC", EC_min, EC_fraction, verbosity);
  CounterChecker BCChecker(CounterChecker::BCChecker, "BC", BC_min, BC_fraction, verbosity);

  // initialise structure merging vfat frame data with the mapping
  for (auto &p : mapping.VFATMapping)
  {
    TotemVFATStatus st;
    st.setMissing(true);
    records[p.first] = { &p.second, NULL,  st };
  }

  // event error message buffer
  stringstream ees;

  // associate data frames with records
  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next())
  {
    // frame error message buffer
    stringstream fes;

    bool problemsPresent = false;
    bool stopProcessing = false;
    
    // skip data frames not listed in the DAQ mapping
    auto records_it = records.find(fr.Position());
    if (records_it == records.end())
    {
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
    if (testFootprint != tfNoTest && !record.frame->checkFootprint())
    {
      problemsPresent = true;
  
      if (verbosity > 0)
        fes << "    invalid footprint\n";

      if ((testFootprint == tfErr))
      {
        record.status.setFootprintError();
        stopProcessing = true;
      }
    }
    
    // check CRC
    if (testCRC != tfNoTest && !record.frame->checkCRC())
    {
      problemsPresent = true;

      if (verbosity > 0)
        fes << "    CRC failure\n";

      if (testCRC == tfErr)
      {
        record.status.setCRCError();
        stopProcessing = true;
      }
    }

    // check the id mismatch
    if (testID != tfNoTest && record.frame->isIDPresent() && (record.frame->getChipID() & 0xFFF) != (record.info->hwID & 0xFFF))
    {
      problemsPresent = true;

      if (verbosity > 0)
        fes << "    ID mismatch (data: 0x" << hex << record.frame->getChipID()
          << ", mapping: 0x" << record.info->hwID  << dec << ", symbId: " << record.info->symbolicID.symbolicID << ")\n";

      if (testID == tfErr)
      {
        record.status.setIDMismatch();
        stopProcessing = true;
      }
    }

    // if there were errors, put the information to ees buffer
    if (verbosity > 0 && problemsPresent)
    {
      string message = (stopProcessing) ? "(and will be dropped)" : "(but will be used though)";
      if (verbosity > 2)
      {
        ees << "  Frame at " << fr.Position() << " seems corrupted " << message << ":\n";
        ees << fes.rdbuf();
      } else
        ees << "  Frame at " << fr.Position() << " seems corrupted " << message << ".\n";
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
  if (verbosity > 1)
  {
    for (const auto &p : records)
    {
      if (p.second.status.isMissing())
        ees << "Frame for VFAT " << p.first << " is not present in the data.\n"; 
    }
  }

  // print error message
  if (verbosity > 0 && !ees.rdbuf()->str().empty())
  {
    if (verbosity > 1)
      LogProblem("Totem") << "Error in RawToDigiConverter::RunCommon > " << "event contains the following problems:\n" << ees.rdbuf() << endl;
    else
      LogProblem("Totem") << "Error in RawToDigiConverter::RunCommon > " << "event contains problems." << endl;
  }

  // increase error counters
  if (printErrorSummary)
  {
    for (const auto &it : records)
    {
      if (!it.second.status.isOK())
      {
        auto &m = errorSummary[it.first];
        m[it.second.status]++;
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::Run(const VFATFrameCollection &input,
  const TotemDAQMapping &mapping, const TotemAnalysisMask &analysisMask,
  DetSetVector<TotemRPDigi> &rpData, DetSetVector<TotemVFATStatus> &finalStatus)
{
  // structure merging vfat frame data with the mapping
  map<TotemFramePosition, Record> records;

  // common processing - frame validation
  RunCommon(input, mapping, records);

  // second loop over data
  for (auto &p : records)
  {
    Record &record = p.second;

    // check whether the data come from RP VFATs
    if (record.info->symbolicID.subSystem != TotemSymbID::RP)
    {
      LogProblem("Totem") << "Error in RawToDigiConverter::Run > "
        << "VFAT is not from RP. subSystem = " << record.info->symbolicID.subSystem;
      continue;
    }

    // silently ignore RP CC VFATs
    if (record.info->type != TotemVFATInfo::data)
      continue;

    // calculate ids
    unsigned short chipId = record.info->symbolicID.symbolicID;
    det_id_type detId = TotemRPDetId::decToRawId(chipId / 10);
    uint8_t chipPosition = chipId % 10;

    // update chipPosition in status
    record.status.setChipPosition(chipPosition);

    // produce digi only for good frames
    if (record.status.isOK())
    {
      // find analysis mask (needs a default=no mask, if not in present the mapping)
      TotemVFATAnalysisMask anMa;
      anMa.fullMask = false;
  
      auto analysisIter = analysisMask.analysisMask.find(record.info->symbolicID);
      if (analysisIter != analysisMask.analysisMask.end())
      {            
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
    
      for (auto ch : activeChannels)
      {
        // skip masked channels
        if (!anMa.fullMask && anMa.maskedChannels.find(ch) == anMa.maskedChannels.end())
        {
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

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::PrintSummaries()
{
  if (printErrorSummary)
  {
    LogVerbatim("Totem") << "* Error summary (error signature : number of such events)" << endl;
    for (const auto &vit : errorSummary)
    {
      LogVerbatim("Totem") << vit.first << endl;

      for (const auto &it : vit.second)
        LogVerbatim("Totem") << "    " << it.first << " : " << it.second << endl;
    }
  }

  if (printUnknownFrameSummary)
  {
    LogVerbatim("Totem") << "* Frames found in data, but not in the mapping (frame position : number of events)" << endl;
    for (const auto &it : unknownSummary)
      LogVerbatim("Totem") << "  " << it.first << " : " << it.second << endl;
  }
}
