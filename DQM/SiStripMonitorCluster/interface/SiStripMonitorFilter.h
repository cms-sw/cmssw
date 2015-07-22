#ifndef SiStripMonitorCluster_SiStripMonitorFilter_h
#define SiStripMonitorCluster_SiStripMonitorFilter_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorFilter
// Original Author: dkcira


// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class DQMStore;

class SiStripMonitorFilter : public DQMEDAnalyzer {
   public:
      explicit SiStripMonitorFilter(const edm::ParameterSet&);
      ~SiStripMonitorFilter(){};

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

   private:
       edm::EDGetTokenT<int> filerDecisionToken_;       
       DQMStore* dqmStore_;
       edm::ParameterSet conf_;
       MonitorElement * FilterDecision;
       // all events
       std::string FilterDirectory;
};

#endif
