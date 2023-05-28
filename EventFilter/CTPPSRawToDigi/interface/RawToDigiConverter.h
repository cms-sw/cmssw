/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Nicola Minafra
*   Laurent Forthomme
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_RawToDigiConverter
#define EventFilter_CTPPSRawToDigi_RawToDigiConverter

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"

#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"

/// \brief Collection of code to convert TOTEM raw data into digi.
class RawToDigiConverter {
public:
  RawToDigiConverter(const edm::ParameterSet &conf);

  /// Creates RP digi.
  void run(const VFATFrameCollection &coll,
           const TotemDAQMapping &mapping,
           const TotemAnalysisMask &mask,
           edm::DetSetVector<TotemRPDigi> &digi,
           edm::DetSetVector<TotemVFATStatus> &status);

  /// Creates Diamond digi.
  void run(const VFATFrameCollection &coll,
           const TotemDAQMapping &mapping,
           const TotemAnalysisMask &mask,
           edm::DetSetVector<CTPPSDiamondDigi> &digi,
           edm::DetSetVector<TotemVFATStatus> &status);

  /// Creates Totem Timing digi.
  void run(const VFATFrameCollection &coll,
           const TotemDAQMapping &mapping,
           const TotemAnalysisMask &mask,
           edm::DetSetVector<TotemTimingDigi> &digi,
           edm::DetSetVector<TotemVFATStatus> &status);

  /// Creates Totem T2 digi
  void run(const VFATFrameCollection &coll,
           const TotemDAQMapping &mapping,
           const TotemAnalysisMask &mask,
           edmNew::DetSetVector<TotemT2Digi> &digi,
           edm::DetSetVector<TotemVFATStatus> &status);

  /// Print error summaries.
  void printSummaries() const;

private:
  struct Record {
    const TotemVFATInfo *info;
    const VFATFrame *frame;
    TotemVFATStatus status;
  };

  const unsigned char verbosity;

  const bool printErrorSummary;
  const bool printUnknownFrameSummary;

  enum TestFlag { tfNoTest, tfWarn, tfErr };

  const bool olderTotemT2FileTest;  //Test file with T2 frame ver 2.1

  /// flags for which tests to run
  const unsigned int testFootprint;
  const unsigned int testCRC;
  const unsigned int testID;
  const unsigned int testECMostFrequent;
  const unsigned int testBCMostFrequent;

  /// the minimal required number of frames to determine the most frequent counter value
  const unsigned int EC_min, BC_min;

  /// the minimal required (relative) occupancy of the most frequent counter value to be accepted
  const double EC_fraction, BC_fraction;

  /// error summaries
  std::map<TotemFramePosition, std::map<TotemVFATStatus, unsigned int> > errorSummary;
  std::map<TotemFramePosition, unsigned int> unknownSummary;

  /// Common processing for all VFAT based sub-systems.
  void runCommon(const VFATFrameCollection &input,
                 const TotemDAQMapping &mapping,
                 std::map<TotemFramePosition, Record> &records);
};

#endif
