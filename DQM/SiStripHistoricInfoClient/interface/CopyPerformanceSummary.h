// -*- C++ -*-
// Package:    DQM/SiStripHistoricInfoClient
// Class:      CopyPerformanceSummary
/**\class CopyPerformanceSummary CopyPerformanceSummary.cc DQM/SiStripHistoricInfoClient/src/CopyPerformanceSummary.cc
 Description: <one line class summary>
 Implementation:
     <Data Quality Monitoring client for long-term detector performance of the Silicon Strip Tracker>
*/
// Original Author:  Dorian Kcira
//         Created:  Wed Apr 25 05:10:12 CEST 2007
// $Id: CopyPerformanceSummary.h,v 1.2 2008/02/15 15:05:36 dutta Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"

namespace edm {
    class ParameterSet;
    class Event;
    class EventId;
    class Timestamp;
}

class CopyPerformanceSummary : public edm::EDAnalyzer {
   public:
      explicit CopyPerformanceSummary(const edm::ParameterSet&);
      ~CopyPerformanceSummary();
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void writeToDB(const edm::Run& run) const;
   private:
      int nevents;
      bool firstEventInRun;
      SiStripPerformanceSummary* pSummary_;
};

