// -*- C++ -*-
//
// Package:    LogMessageMonitor
// Class:      LogMessageMonitor
// 
/**\class LogMessageMonitor LogMessageMonitor.cc DQM/LogMonitor/src/LogMessageMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mia Tosi,40 3-B32,+41227671609,
//         Created:  Thu Mar  8 14:34:13 CET 2012
// $Id: LogMessageMonitor.h,v 1.1 2012/10/15 13:24:45 threus Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>
#include <string>
#include <map>

class DQMStore;
class GenericTriggerEventFlag;

class GetLumi;

//
// class declaration
//

class LogMessageMonitor : public edm::EDAnalyzer {
   public:
      explicit LogMessageMonitor(const edm::ParameterSet&);
      ~LogMessageMonitor();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
      std::string histname;  //for naming the histograms according to algorithm used
      
      DQMStore * dqmStore_;
      edm::ParameterSet conf_;

      std::map<std::string,int> modulesMap;

      // from parameters
  std::string pluginsMonName_;
  std::vector<std::string> modules_vector_;
  std::vector<std::string> categories_vector_;
  
  GetLumi* lumiDetails_;
  GenericTriggerEventFlag* genTriggerEventFlag_;
  
  // MEs
  std::vector<MonitorElement*> ModulesErrorsVsBXlumi;
  std::vector<MonitorElement*> ModulesWarningsVsBXlumi;
  
  MonitorElement* CategoriesVsModules;

  bool doWarningsPlots_;
  bool doPUmonitoring_;
      
};
