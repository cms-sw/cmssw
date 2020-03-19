#ifndef CSCObjects_CSCGainsStudy_h
#define CSCObjects_CSCGainsStudy_h

/** \class CSCGainsStudy
 * 
 * Reads in CSC database and computes average strip gain "G" for whole CSC system.
 * For each strip i, it computes a factor G/g_i, where g_i is the gain for strip i.
 * The correction factors are then plotted for each chamber.
 *
 * \author Dominique Fortin - UCR
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

#include <CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMappingForSliceTest.h>

#include "CSCGainsStudyHistograms.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TFile;

class CSCGainsStudy : public edm::EDAnalyzer {
public:
  /// configurable parameters
  explicit CSCGainsStudy(const edm::ParameterSet& p);

  ~CSCGainsStudy();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

  /// Compute average Gain for all CSC chambers
  float getStripGainAvg();

private:
  // Pointers to histograms
  HCSCGains* All_CSC;
  // ME+1/1
  HCSCGains* ME_11_27;
  HCSCGains* ME_11_28;
  HCSCGains* ME_11_29;
  HCSCGains* ME_11_30;
  HCSCGains* ME_11_31;
  HCSCGains* ME_11_32;
  // ME+1/2
  HCSCGains* ME_12_27;
  HCSCGains* ME_12_28;
  HCSCGains* ME_12_29;
  HCSCGains* ME_12_30;
  HCSCGains* ME_12_31;
  HCSCGains* ME_12_32;
  // ME+1/3
  HCSCGains* ME_13_27;
  HCSCGains* ME_13_28;
  HCSCGains* ME_13_29;
  HCSCGains* ME_13_30;
  HCSCGains* ME_13_31;
  HCSCGains* ME_13_32;
  // ME+2/1
  HCSCGains* ME_21_14;
  HCSCGains* ME_21_15;
  HCSCGains* ME_21_16;
  // ME+2/2
  HCSCGains* ME_22_27;
  HCSCGains* ME_22_28;
  HCSCGains* ME_22_29;
  HCSCGains* ME_22_30;
  HCSCGains* ME_22_31;
  HCSCGains* ME_22_32;
  // ME+3/1
  HCSCGains* ME_31_14;
  HCSCGains* ME_31_15;
  HCSCGains* ME_31_16;
  // ME+3/2
  HCSCGains* ME_32_27;
  HCSCGains* ME_32_28;
  HCSCGains* ME_32_29;
  HCSCGains* ME_32_30;
  HCSCGains* ME_32_31;
  HCSCGains* ME_32_32;

  // Mapping file stuff
  std::string CSCMapFile;
  CSCReadoutMappingFromFile theCSCMap;

  // The file which will store the histos
  TFile* theFile;

  // Switch for debug output
  bool debug;

  // Root file name
  std::string rootFileName;

  // Store in memory the Gains
  const CSCGains* pGains;
};

#endif
