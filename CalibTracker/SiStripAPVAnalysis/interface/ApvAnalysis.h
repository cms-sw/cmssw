#ifndef Tracker_APVAnalysis_h
#define Tracker_APVAnalysis_h

#include <vector>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "FWCore/Framework/interface/Event.h"

//#define DEBUG_INSTANCE_COUNTING

#ifdef DEBUG_INSTANCE_COUNTING
#include "CommonDet/DetUtilities/interface/InstanceCounting.h"
#endif

class TkApvMask;
class TkCommonModeCalculator;
class TkPedestalCalculator;
class TkNoiseCalculator;

#include <vector>
#include <utility>

/**
 * ApvAnalysis is the base class for the simulation of APV/FED.
 * Each instance has
 * - a TkPedestalCalculator
 * - a TkCommonModeCalculator
 * - a TkApvMask
 * - a TkNoiseCalculator
 */

class ApvAnalysis
#ifdef DEBUG_INSTANCE_COUNTING
    : public InstanceCounting<ApvAnalysis>
#endif
{
public:
  typedef edm::DetSet<SiStripRawDigi> RawSignalType;
  typedef std::vector<float> PedestalType;

  ApvAnalysis(int nev);
  ~ApvAnalysis() { ; }

  //
  // Tell ApvAnalysis which algorithms to use.
  //

  void setCommonModeCalculator(TkCommonModeCalculator& in) { theTkCommonModeCalculator = &in; }
  void setPedestalCalculator(TkPedestalCalculator& in) { theTkPedestalCalculator = &in; }
  void setNoiseCalculator(TkNoiseCalculator& in) { theTkNoiseCalculator = &in; }
  void setMask(TkApvMask& in) { theTkApvMask = &in; }

  TkCommonModeCalculator& commonModeCalculator() { return *theTkCommonModeCalculator; }
  TkPedestalCalculator& pedestalCalculator() { return *theTkPedestalCalculator; }
  TkNoiseCalculator& noiseCalculator() { return *theTkNoiseCalculator; }
  TkApvMask& mask() { return *theTkApvMask; }

  //
  // Give store/load commands to the TkPedestalCalculator and to
  // TkNoiseCalculator, will use ApvEventReader.
  // Has to be done here because they have no access to the reader.
  //
  /** Update pedestals & noise with current event */
  void updateCalibration(edm::DetSet<SiStripRawDigi>& in);
  void newEvent() const;

private:
  TkCommonModeCalculator* theTkCommonModeCalculator;
  TkPedestalCalculator* theTkPedestalCalculator;
  TkNoiseCalculator* theTkNoiseCalculator;
  TkApvMask* theTkApvMask;
  int nEventsForNoiseCalibration_;
  int eventsRequiredToUpdate_;
};
#endif
