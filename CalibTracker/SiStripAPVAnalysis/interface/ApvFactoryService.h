#ifndef ApvAnalysis_ApvFactoryService_H
#define ApvAnalysis_ApvFactoryService_H

#define DATABASE  //@@ necessary?

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <string>
#include <map>

/**	
   \class ApvFactoryService
   \brief An interface class to set the parameter in ApvAnalysisFactory  
*/

class ApvFactoryService {
public:
  // -------------------- Constructors, destructors --------------------

  /** Constructor when using the "service" mode, which takes as an
      argument a ParameterSet (containing the database connection
      parameters). */
  ApvFactoryService(const edm::ParameterSet&, edm::ActivityRegistry&);

  /** Default destructor. */
  ~ApvFactoryService();

  int getEventInitNumber();

  ApvAnalysisFactory* const getApvFactory() const;

  // -------------------- Structs and enums --------------------

  void postProcessEvent(const edm::Event& ie, const edm::EventSetup& ies);

  /** Class that holds addresses that uniquely identify a hardware
      component within the control system. */

private:
  // -------------------- Miscellaneous private methods --------------------

  /** Instance of struct that holds all DB connection parameters. */
  ApvAnalysisFactory* apvFactory_;

  bool gotPed;
};

#endif  // ApvFactoryService_H
