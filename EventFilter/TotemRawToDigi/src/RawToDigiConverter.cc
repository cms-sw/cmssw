/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/RawToDigiConverter.h"

#include "EventFilter/TotemRawToDigi/interface/CounterChecker.h"

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
      map<TotemFramePosition, TotemVFATStatus> &status)
{
  // map which will contain FramePositions from mapping missing in raw event
  map<TotemFramePosition, TotemVFATInfo> missingFrames(mapping.VFATMapping);

  // EC and BC checks (wrt. the most frequent value), BC checks per subsystem
  CounterChecker ECChecker(CounterChecker::ECChecker, "EC", EC_min, EC_fraction, verbosity);
  CounterChecker BCChecker(CounterChecker::BCChecker, "BC", BC_min, BC_fraction, verbosity);

  // loop over data frames
  stringstream ees;

  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next())
  {
    stringstream fes;
    bool stopProcessing = false;
    
    // skip data frames not listed in the DAQ mapping
    auto mappingIter = mapping.VFATMapping.find(fr.Position());
    if (mappingIter == mapping.VFATMapping.end())
    {
      unknownSummary[fr.Position()]++;
      continue;
    }

    // contain information about processed frame
    TotemVFATStatus &actualStatus = status[fr.Position()];

    // remove from missingFrames
    auto iter = missingFrames.find(fr.Position());
    missingFrames.erase(iter);

    // check footprint
    if (testFootprint != tfNoTest && !fr.Data()->checkFootprint())
    {
      fes << "    invalid footprint\n";
      if ((testFootprint == tfErr))
      {
        actualStatus.setFootprintError();
        stopProcessing = true;
      }
    }
    
    // check CRC
    if (testCRC != tfNoTest && !fr.Data()->checkCRC())
    {
      fes << "    CRC failure\n";
      if (testCRC == tfErr)
      {
        actualStatus.setCRCError();
        stopProcessing = true;
      }
    }

    // check the id mismatch
    if (testID != tfNoTest && fr.Data()->isIDPresent() && (fr.Data()->getChipID() & 0xFFF) != (mappingIter->second.hwID & 0xFFF))
    {
      fes << "    ID mismatch (data: 0x" << hex << fr.Data()->getChipID()
        << ", mapping: 0x" << mappingIter->second.hwID  << dec << ", symbId: " << mappingIter->second.symbolicID.symbolicID << ")\n";
      if (testID == tfErr)
      {
        actualStatus.setIDMismatch();
        stopProcessing = true;
      }
    }

    // if there were errors, put the information to ees buffer
    if (verbosity > 0 && !fes.rdbuf()->str().empty())
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
    ECChecker.Analyze(status, (testECMostFrequent == tfErr), ees);

  if (testBCMostFrequent != tfNoTest)
    BCChecker.Analyze(status, (testBCMostFrequent == tfErr), ees);

  // save the information about missing frames to conversionStatus
  for (const auto &it : missingFrames)
  {
    TotemVFATStatus &actualStatus = status[it.first];
    actualStatus.setMissing();
    if (verbosity > 1) 
      ees << "Frame for VFAT " << it.first << " is not present in the data.\n";
  }

  // print error message
  if (verbosity > 0 && !ees.rdbuf()->str().empty())
  {
    if (verbosity > 1)
      cerr << "event contains the following problems:\n" << ees.rdbuf() << endl;
    else
      cerr << "event contains problems." << endl;
  }

  // increase error counters
  if (printErrorSummary)
  {
    for (const auto &it : status)
    {
      if (!it.second.isOK())
      {
        auto &m = errorSummary[it.first];
        m[it.second]++;
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::Run(const VFATFrameCollection &input,
  const TotemDAQMapping &mapping, const TotemAnalysisMask &analysisMask,
  DetSetVector<TotemRPDigi> &rpData, DetSetVector<TotemVFATStatus> &finalStatus)
{
  // map to link TotemFramePosition to the conversion status
  map<TotemFramePosition, TotemVFATStatus> status;

  // common processing - frame validation
  RunCommon(input, mapping, status);

  // second loop over data
  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next())
  {
    auto mappingIter = mapping.VFATMapping.find(fr.Position());
    if (mappingIter == mapping.VFATMapping.end())
      continue;
  
    // check whether the data come from RP VFATs
    if (mappingIter->second.symbolicID.subSystem != TotemSymbID::RP)
      throw cms::Exception("RawToDigiConverter::Run") << "VFAT is not from RP. subSystem = " <<
        mappingIter->second.symbolicID.subSystem << endl;

    // silently ignore RP CC VFATs
    if (mappingIter->second.type != TotemVFATInfo::data)
      continue;

    // calculate ids
    unsigned short chipId = mappingIter->second.symbolicID.symbolicID;
    det_id_type detId = TotemRPDetId::decToRawId(chipId / 10);
    uint8_t chipPosition = chipId % 10;

    // update chipPosition in status
    TotemVFATStatus &actualStatus = status[fr.Position()];
    actualStatus.setChipPosition(chipPosition);

    // produce digi only for good frames
    if (actualStatus.isOK())
    {
      // find analysis mask (needs a default=no mask, if not in present the mapping)
      TotemVFATAnalysisMask anMa;
      anMa.fullMask = false;
  
      auto analysisIter = analysisMask.analysisMask.find(mappingIter->second.symbolicID);
      if (analysisIter != analysisMask.analysisMask.end())
      {            
        // if there is some information about masked channels - save it into conversionStatus
        anMa = analysisIter->second;
        if (anMa.fullMask)
          actualStatus.setFullyMaskedOut();
        else
          actualStatus.setPartiallyMaskedOut();
      }
  
      // create the digi
      unsigned short offset = chipPosition * 128;
      const vector<unsigned char> &activeChannels = fr.Data()->getActiveChannels();
    
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
    statusDetSet.push_back(actualStatus);
  }
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::PrintSummaries()
{
  if (printErrorSummary)
  {
    cout << "* Error summary (error signature : number of such events)" << endl;
    for (const auto &vit : errorSummary)
    {
      cout << vit.first << endl;

      for (const auto &it : vit.second)
        cout << "    " << it.first << " : " << it.second << endl;
    }
  }

  if (printUnknownFrameSummary)
  {
    cout << "* Frames found in data, but not in the mapping (frame position : number of events)" << endl;
    for (const auto &it : unknownSummary)
      cout << "  " << it.first << " : " << it.second << endl;
  }
}
