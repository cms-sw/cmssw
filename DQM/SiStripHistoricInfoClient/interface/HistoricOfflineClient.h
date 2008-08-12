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
// $Id: HistoricOfflineClient.h,v 1.2 2008/02/15 15:05:36 dutta Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
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
      void fillSummaryObjects(const edm::Run& run) const;
      void writeToDB(edm::EventID evid, edm::Timestamp evtime) const;
      void writeToDB(const edm::Run& run) const;
      float CalculatePercentOver(MonitorElement * me) const;
   private:
      int nevents;
      bool firstEventInRun;
      edm::ParameterSet parameters;
      DQMStore* dqmStore_;
      std::map<uint32_t, std::vector<MonitorElement *> > ClientPointersToModuleMEs;
      SiStripPerformanceSummary* pSummary_;
};

