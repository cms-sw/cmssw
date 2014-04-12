#ifndef HcalLaserEventFilter2012_h
#define HcalLaserEventFilter2012_h

// -*- C++ -*-
//
// Package:    HcalLaserEventFilter2012
// Class:      HcalLaserEventFilter2012
// 
/**\class HcalLaserEventFilter2012 HcalLaserEventFilter2012.cc UserCode/HcalLaserEventFilter2012/src/HcalLaserEventFilter2012.cc

 Description: [Remove known HCAL laser events in 2012 data]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jeff Temple, University of Maryland (jtemple@fnal.gov)
//         Created:  Fri Oct 19 13:15:44 EDT 2012
//
//


// system include files
#include <iostream>
#include <sstream>
#include <fstream>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//
class HcalLaserEventFiltProducer2012;

class HcalLaserEventFilter2012 : public edm::EDFilter {
public:
  explicit HcalLaserEventFilter2012(const edm::ParameterSet&);
  ~HcalLaserEventFilter2012();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  friend HcalLaserEventFiltProducer2012;

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  
  void readEventListFile(const std::string & eventFileName);
  void addEventString(const std::string & eventString);

 // ----------member data ---------------------------
  typedef std::vector< std::string > strVec;
  typedef std::vector< std::string >::iterator strVecI;
  
  std::vector< std::string > EventList_;  // vector of strings representing bad events, with each string in "run:LS:event" format
  bool verbose_;  // if set to true, then the run:LS:event for any event failing the cut will be printed out
  std::string prefix_;  // prefix will be printed before any event if verbose mode is true, in order to make searching for events easier
  
  // Set run range of events in the BAD LASER LIST.  
  // The purpose of these values is to shorten the length of the EventList_ vector when running on only a subset of data
  int minrun_;
  int maxrun_;  // if specified (i.e., values > -1), then only events in the given range will be filtered
  int minRunInFile, maxRunInFile;

  bool WriteBadToFile_;
  bool forceFilterTrue_;
  std::ofstream outfile_;
};
#endif
