#ifndef DQMOffline_L1Trigger_L1TBeamConfiguration_h
#define DQMOffline_L1Trigger_L1TBeamConfiguration_h

/*
 * \class L1TBeamConfiguration
 *
 *
 * Description: offline DQM class for acquiring beam configuration
 *
 * Implementation:
 *   <TODO: enter implementation details>
 *
 * \author: Pietro Vischia - LIP Lisbon pietro.vischia@gmail.com
 *
 * Changelog:
 *    2012/11/22 12:01:01: Creation, infrastructure and generic crap
 *
 * Todo:
 *  -
 *  -
 *
 * $Date: 2012/11/27 14:56:18 $
 * $Revision: 1.1 $
 *
 */

// System include files
//#include <memory>
//#include <unistd.h>

// User include files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/LuminosityBlock.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
//#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQMServices/Core/interface/MonitorElement.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"
//
//#include <TString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

// Forward declarations



// Helper

// Class declaration


class L1TBeamConfiguration{

 public:

  L1TBeamConfiguration();

  bool bxConfig(unsigned iBx);

  bool isValid(){return m_valid;}

  bool m_valid;           // Bit Name for which the fit refers to
  std::vector<bool> beam1;
  std::vector<bool> beam2;

};


#endif

