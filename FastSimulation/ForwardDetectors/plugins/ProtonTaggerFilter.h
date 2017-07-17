#ifndef FastSimulation_ProtonTaggerFilter_H
#define FastSimulation_ProtonTaggerFilter_H

/// Fast simulation of near-beam detector acceptance.

/**
 * This class defines an EDFilter which does the following:
 * - reads generated data (edm::HepMCProduct) from edm::Event
 * - selects forward protons
 * - determines (by means of AcceptanceTableHelper) the acceptance
 *   of near-beam detectors: FP420 and TOTEM
 * - returns a boolean value representing whether the proton(s) were seen
 *   with certain detectors, several options are available to choose
 *   between FP420/TOTEM detectors and their combinations
 *
 * Originally this code was meant as the FastTotem module from ORCA-based FAMOS,
 * ported to the CMSSW framework. However it was eventually re-written from scratch.
 * Nevelrtheless, the physics performace is just the same as one of FastTotem,
 * as it (currently) uses the same acceptance tables.
 * 
 * Author: Dmitry Zaborov
 */

// Version: $Id: ProtonTaggerFilter.h,v 1.1 2008/11/25 17:34:15 beaudett Exp $

#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FastSimulation/ForwardDetectors/plugins/AcceptanceTableHelper.h"

#include "HepMC/GenEvent.h"

#include "TFile.h"

class ProtonTaggerFilter : public edm::stream::EDFilter <>
{

 public:

  /// default constructor
  explicit ProtonTaggerFilter(edm::ParameterSet const & p);

  /// empty destructor
  virtual ~ProtonTaggerFilter();

  /// startup function of the EDFilter
  virtual void beginJob();

  /// endjob function of the EDFilter
  virtual void endJob();

  /// decide if the event is accepted by the proton taggers
  virtual bool filter(edm::Event & e, const edm::EventSetup & c);

 private:

  /// choose which of the detectors (FP420/TOTEM/both) will be used for beam 1
  unsigned int beam1mode;

  /// choose which of the detectors (FP420/TOTEM/both) will be used for beam 2
  unsigned int beam2mode;

  /// choose how to combine data from the two beams (ask for 1/2 proton)
  unsigned int beamCombiningMode;

  /// Objects which actually compute the acceptance (one per detector or combination of detectors)
  AcceptanceTableHelper helper420beam1;
  AcceptanceTableHelper helper420beam2;
  AcceptanceTableHelper helper220beam1;
  AcceptanceTableHelper helper220beam2;
  AcceptanceTableHelper helper420a220beam1;
  AcceptanceTableHelper helper420a220beam2;
  
};

#endif
