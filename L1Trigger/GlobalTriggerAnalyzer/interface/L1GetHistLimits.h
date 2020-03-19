#ifndef L1Trigger_GlobalTriggerAnalyzer_L1GetHistLimits_h
#define L1Trigger_GlobalTriggerAnalyzer_L1GetHistLimits_h

/**
 * \class L1GetHistLimits
 *
 *
 * Description: use L1 scales to define histogram limits for L1 trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// system include files
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>

// user include files
//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"

// scales
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

// forward declarations

// class declaration
class L1GetHistLimits {
public:
  // constructor(s)
  explicit L1GetHistLimits(const edm::EventSetup& evSetup);

  // destructor
  virtual ~L1GetHistLimits();

  /// structure containing all limits:
  /// numbers of bins for a given histogram
  /// lower limit of the first bin in the histogram
  /// upper limit of the last bin in the histogram
  /// vector of bin thresholds
  struct L1HistLimits {
    int nrBins;
    double lowerBinValue;
    double upperBinValue;
    std::vector<float> binThresholds;
  };

public:
  /// for a L1 trigger object and a given quantity,
  /// return all limits for a histogram
  const L1HistLimits& l1HistLimits(const L1GtObject& l1GtObject, const std::string& quantity);

  /// for a L1 trigger object and a given quantity,
  /// return the real limits for a histogram given an arbitrary range
  const L1HistLimits& l1HistLimits(const L1GtObject& l1GtObject,
                                   const std::string& quantity,
                                   const double histMinValue,
                                   const double histMaxValue);

  /// for a L1 trigger object and a given quantity,
  /// return the numbers of bins for a given histogram
  const int l1HistNrBins(const L1GtObject& l1GtObject, const std::string& quantity);

  /// for a L1 trigger object and a given quantity,
  /// return the lower limit of the first bin in the histogram
  const double l1HistLowerBinValue(const L1GtObject& l1GtObject, const std::string& quantity);

  /// for a L1 trigger object and a given quantity,
  /// return the upper limit of the last bin in the histogram
  const double l1HistUpperBinValue(const L1GtObject& l1GtObject, const std::string& quantity);

  /// for a L1 trigger object and a given quantity,
  /// return the vector of bin thresholds
  const std::vector<float>& l1HistBinThresholds(const L1GtObject& l1GtObject, const std::string& quantity);

private:
  /// for a L1 trigger object and a given quantity,
  /// compute the number of bins, the lower limit of the first bin,
  /// the upper limit of the last bin and the vector of bin thresholds
  void getHistLimits(const L1GtObject& l1GtObject, const std::string& quantity);

private:
  const edm::EventSetup& m_evSetup;

  /// all limits for a histogram
  L1HistLimits m_l1HistLimits;
};

#endif
