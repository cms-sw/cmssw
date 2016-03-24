/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/RawToDigiConverter.h"

#include "EventFilter/TotemRawToDigi/interface/CounterChecker.h"

#include "DataFormats/TotemRPDetId/interface/TotRPDetId.h"

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

int RawToDigiConverter::Run(const VFATFrameCollection &input,
  const TotemDAQMapping &mapping, const TotemAnalysisMask &analysisMask,
  edm::DetSetVector<TotemRPDigi> &rpData, std::vector <TotemRPCCBits> &rpCC, TotemRawToDigiStatus &status)
{
  // map which will contain FramePositions from mapping, which data is missing in raw event
  map<TotemFramePosition, TotemVFATInfo> missingFrames(mapping.VFATMapping);

  // EC and BC checks (wrt. the most frequent value), BC checks per subsystem
  CounterChecker ECChecker(CounterChecker::ECChecker, "EC", EC_min, EC_fraction, verbosity);
  map<unsigned int, CounterChecker> BCCheckers;
  BCCheckers[TotemSymbID::RP] = CounterChecker(CounterChecker::BCChecker, "BC/RP", BC_min, BC_fraction, verbosity);

  // loop over data frames
  stringstream ees;

  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next())
  {
    stringstream fes;
    bool stopProcessing = false;

    // contain information about processed frame
    TotemVFATStatus &actualStatus = status[fr.Position()];
    
    // skip unlisted positions (TotemVFATInfo)
    auto mappingIter = mapping.VFATMapping.find(fr.Position());
    if (mappingIter == mapping.VFATMapping.end())
    {
      unknownSummary[fr.Position()]++;
      continue;
    }

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
      BCCheckers[mappingIter->second.symbolicID.subSystem].Fill(fr.Data()->getBC(), fr.Position());
  }

  // analyze EC and BC statistics
  if (testECMostFrequent != tfNoTest)
    ECChecker.Analyze(status, (testECMostFrequent == tfErr), ees);

  if (testBCMostFrequent != tfNoTest)
  {
    for (map<unsigned int, CounterChecker>::iterator it = BCCheckers.begin(); it != BCCheckers.end(); ++it)
      it->second.Analyze(status, (testBCMostFrequent == tfErr), ees);
  }

  // save the information about missing frames to conversionStatus
  for (map<TotemFramePosition, TotemVFATInfo>::iterator iter = missingFrames.begin(); iter != missingFrames.end(); iter++)
  {
    TotemVFATStatus &actualStatus = status[iter->first];
    actualStatus.setMissing();
    if (verbosity > 1) 
      ees << "Frame for VFAT " << iter->first << " is not present in the data.\n";
  }

  // print error message
  if (verbosity > 0 && !ees.rdbuf()->str().empty())
  {
    if (verbosity > 1)
      cerr << "event contains the following problems:\n" << ees.rdbuf() << endl;
    else
      cerr << "event contains problems." << endl;
  }

  // update error summary
  if (printErrorSummary)
  {
    for (TotemRawToDigiStatus::iterator it = status.begin(); it != status.end(); ++it)
    {
      if (!it->second.OK())
      {
        map<TotemVFATStatus, unsigned int> &m = errorSummary[it->first];
        m[it->second]++;
      }
    }
  }

  // produce digi for good frames
  for (VFATFrameCollection::Iterator fr(&input); !fr.IsEnd(); fr.Next())
  {
    map<TotemFramePosition, TotemVFATInfo>::const_iterator mappingIter = mapping.VFATMapping.find(fr.Position());
    if (mappingIter == mapping.VFATMapping.end())
      continue;

    TotemVFATStatus &actualStatus = status[fr.Position()];

    if (!actualStatus.OK())
      continue;

    // prepare analysis mask class
    auto analysisIter = analysisMask.analysisMask.find(mappingIter->second.symbolicID);

    // find analysis mask
    TotemVFATAnalysisMask anMa;
    anMa.fullMask = false;
    if (analysisIter != analysisMask.analysisMask.end())
    {            
      // if there is some information about masked channels - save it into conversionStatus
      anMa = analysisIter->second;
      if (anMa.fullMask)
        actualStatus.setFullyMaskedOut();
      else
        actualStatus.setPartiallyMaskedOut();
    }
    
    // decide which method should process that frame
    switch (mappingIter->second.symbolicID.subSystem)
    {
      case TotemSymbID::RP:  
        switch (mappingIter->second.type)
        {
          case TotemVFATInfo::data:
            RPDataProduce(fr, mappingIter->second, anMa, rpData);    
            break;
          case TotemVFATInfo::CC:
            RPCCProduce(fr, mappingIter->second, anMa, rpCC);
            break;
        }  
        break;

      case TotemSymbID::T1:
        break;

      case TotemSymbID::T2:
        break;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::RPDataProduce(VFATFrameCollection::Iterator &fr, const TotemVFATInfo &info,
    const TotemVFATAnalysisMask &analysisMask, edm::DetSetVector<TotemRPDigi> &rpData)
{
  // get IDs
  unsigned short symId = info.symbolicID.symbolicID;
  unsigned int detId = TotRPDetId::DecToRawId(symId / 10);

  // add TotemRPDigi for each hit
  unsigned short offset = (symId % 10) * 128;
  const vector<unsigned char> activeCh = fr.Data()->getActiveChannels();
  DetSet<TotemRPDigi> &detSet = rpData.find_or_insert(detId);

  for (unsigned int j = 0; j < activeCh.size(); j++)
  {
    // skip masked channels
    if (!analysisMask.fullMask && analysisMask.maskedChannels.find(j) == analysisMask.maskedChannels.end())
    {
      detSet.push_back(TotemRPDigi(detId, offset + activeCh[j]));
    }
  }  
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::RPCCProduce(VFATFrameCollection::Iterator &fr, const TotemVFATInfo &info,
    const TotemVFATAnalysisMask &analysisMask, std::vector <TotemRPCCBits> &rpCC)
{
  // get IDs
  unsigned short symId = info.symbolicID.symbolicID;

  const vector<unsigned char> activeCh = fr.Data()->getActiveChannels();
  
  std::bitset<16> bs_even;
  std::bitset<16> bs_odd;

  bs_even.reset();
  bs_odd.reset();

  unsigned int stripNo;

  // if all channels are masked out, do not process all frame
  if (!analysisMask.fullMask) 
    for (unsigned int j = 0; j < activeCh.size(); j++)
      {
  //      std::cout << "Active channel " << j << " value " << (int)(activeCh[j]) << std::endl;
  // check, whether j channel is not masked out
  if (analysisMask.maskedChannels.find(j) == analysisMask.maskedChannels.end())
    {
      stripNo = (unsigned int) (activeCh[j]);
      unsigned int ch = stripNo + 2; // TODO check if +2 is necessary
      //  std::cout << "Strip no " << (unsigned int)(activeCh[j]) << std::endl;
      //  std::cout << "Channel no " << ch << std::endl;
      if (ch >= 72 && ch <= 100 && (ch % 4 == 0)) 
        {
    bs_even.set(ch/4-18);
    continue;
        }
      if (ch >= 40 && ch <= 68 && (ch % 4 == 0)) 
        {
    bs_even.set(ch/4 - 2);
    continue;
        }
      if (ch == 38) {
        bs_odd.set(15);
        continue;
      }
      if (ch >= 104 && ch <= 128 && (ch % 4 == 0)) 
        {
    bs_odd.set(ch/4 - 18);
    continue;
        }
      if (ch >= 42 && ch <= 70 && (ch % 4 == 2)) 
        {
    bs_odd.set((ch-2)/4 - 9 - 1);
    continue;
        }
    } // end if
      } // end for
  //     std::cout << "Odd " << bs_odd << std::endl;
  //      std::cout << "Even " << bs_even << std::endl;

  unsigned int evendetId = TotRPDetId::DecToRawId(symId * 10);
  unsigned int odddetId = TotRPDetId::DecToRawId(symId * 10 + 1);
  TotemRPCCBits ccbits_even(evendetId , bs_even);
  TotemRPCCBits ccbits_odd(odddetId, bs_odd);
  
  rpCC.push_back(ccbits_even);
  rpCC.push_back(ccbits_odd);
}

//----------------------------------------------------------------------------------------------------

void RawToDigiConverter::PrintSummaries()
{
  if (printErrorSummary)
  {
    cout << "* Error summary (error signature: number of such events)" << endl;
    for (map<TotemFramePosition, map<TotemVFATStatus, unsigned int> >::iterator vit = errorSummary.begin();
        vit != errorSummary.end(); ++vit)
    {
      cout << "  " << vit->first << endl;
      for (map<TotemVFATStatus, unsigned int>::iterator it = vit->second.begin(); it != vit->second.end(); ++it)
      {
        cout << "    " << it->first << ": " << it->second << endl;
      }
    }
  }

  if (printUnknownFrameSummary)
  {
    cout << "* Frames found in data, but not in the mapping (frame position: number of events)" << endl;
    for (map<TotemFramePosition, unsigned int>::iterator it = unknownSummary.begin(); it != unknownSummary.end(); ++it)
    {
      cout << "  " << it->first << ":" << it->second << endl;
    }
  }
}
