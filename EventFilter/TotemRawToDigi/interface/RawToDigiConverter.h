/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef _RawToDigiConverter_h_
#define _RawToDigiConverter_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "EventFilter/TotemRawToDigi/interface/VFATFrameCollection.h"

#include "CondFormats/TotemReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemAnalysisMask.h"

#include "DataFormats/TotemRPDigi/interface/TotemRPDigi.h"
#include "DataFormats/TotemRPL1/interface/TotemRPCCBits.h"
#include "DataFormats/TotemRawData/interface/TotemRawEvent.h"
#include "DataFormats/TotemRawData/interface/TotemRawToDigiStatus.h"

//----------------------------------------------------------------------------------------------------

/// \brief Collection of code to convert TOTEM raw data into digi.
class RawToDigiConverter
{
  private:
  unsigned char verbosity;

  unsigned int printErrorSummary;
  unsigned int printUnknownFrameSummary;

  enum TestFlag { tfNoTest, tfWarn, tfErr };

  /// flags for which tests to run
  unsigned int testFootprint;
  unsigned int testCRC;
  unsigned int testID;
  unsigned int testECRaw;
  unsigned int testECDAQ;
  unsigned int testECMostFrequent;
  unsigned int testBCMostFrequent;

  /// the minimal required number of frames to determine the most frequent counter value
  unsigned int EC_min, BC_min;

  /// the minimal required (relative) occupancy of the most frequent counter value to be accepted
  double EC_fraction, BC_fraction;

  /// error summaries
  std::map<TotemFramePosition, std::map<TotemVFATStatus, unsigned int> > errorSummary;
  std::map<TotemFramePosition, unsigned int> unknownSummary;

  public:
    RawToDigiConverter(const edm::ParameterSet &conf);

    /// Converts vfat data in `coll'' into digi.
    int Run(const VFATFrameCollection &coll,
      const TotemDAQMapping &mapping, const TotemAnalysisMask &mask,
      edm::DetSetVector<TotemRPDigi> &rpData, std::vector<TotemRPCCBits> &rpCC, TotemRawToDigiStatus &status);

    /// Produce Digi from one RP data VFAT.
    void RPDataProduce(VFATFrameCollection::Iterator &fr, const TotemVFATInfo &info,
      const TotemVFATAnalysisMask &analysisMask, edm::DetSetVector<TotemRPDigi> &rpData);

    /// Produce Digi from one RP trigger VFAT.
    void RPCCProduce(VFATFrameCollection::Iterator &fr, const TotemVFATInfo &info,
      const TotemVFATAnalysisMask &analysisMask, std::vector <TotemRPCCBits> &rpCC);

    /// Print error summaries.
    void PrintSummaries();
};

#endif
