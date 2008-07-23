// -*- C++ -*-
// Package:    DQM/SiStripHistoricInfoClient
// Class:      HistoricOfflineClient
/**\class HistoricOfflineClient HistoricOfflineClient.cc DQM/HistoricOfflineClient/src/HistoricOfflineClient.cc
 Description: <one line class summary>
 Implementation:
     <Data Quality Monitoring client for long-term detector performance of the Silicon Strip Tracker>
*/
// Original Author:  Dorian Kcira
//         Created:  Wed Apr 25 05:10:12 CEST 2007
//        Modified:  Anne-Catherine Le Bihan 06/2008
// $Id: HistoricOfflineClient.h,v 1.3 2008/03/02 00:07:41 dutta Exp $

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;
class MonitorElement;

namespace edm {
    class ParameterSet;
    class Event;
    class EventId;
    class Timestamp;
}

class HistoricOfflineClient : public edm::EDAnalyzer {
   public:
      explicit HistoricOfflineClient(const edm::ParameterSet&);
      ~HistoricOfflineClient();
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void retrievePointersToModuleMEs(const edm::EventSetup&);
      void fillSummaryObjects(SiStripSummary* summary,std::string& histoName, std::vector<std::string>& Quantities);
      void writeToDB() const;
      uint32_t returnDetComponent(std::string histoName);

      int nevents;
      bool firstEventInRun;
     
      DQMStore* dqmStore_;
       
      std::map<uint32_t, std::vector<MonitorElement *> > ClientPointersToModuleMEs;
      
      std::vector<SiStripSummary *> vSummary;
     
      edm::ParameterSet iConfig_;
      
};

