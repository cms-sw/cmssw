#ifndef MuonRecoTest_H
#define MuonRecoTest_H


/** \class MuonRecoTest
 * *
 *  DQMOffline Test Client
 *       check the recostruction efficiency of Sta/Glb on eta, phi parameters
 *
 *  \author  G. Mila - INFN Torino
 *   
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>



class MuonRecoTest: public DQMEDHarvester{

public:

  /// Constructor
  MuonRecoTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonRecoTest() {};

protected:

  /// Endjob
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

private:

  // counters
  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  // Switch for verbosity
  std::string metname;
  edm::ParameterSet parameters;

   //histo binning parameters
  std::string EfficiencyCriterionName;
  int etaBin;
  double etaMin;
  double etaMax;

  int phiBin;
  double phiMin;
  double phiMax;

  // efficiency histograms
  MonitorElement* etaEfficiency;
  MonitorElement* phiEfficiency;
  // aligment plot
  std::vector<MonitorElement*> globalRotation;

};

#endif
