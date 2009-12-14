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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
class DQMStore;

class SiStripMonitorFilter : public edm::EDAnalyzer {
   public:
      explicit SiStripMonitorFilter(const edm::ParameterSet&);
      ~SiStripMonitorFilter(){};

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;

   private:
       DQMStore* dqmStore_;
       edm::ParameterSet conf_;
       MonitorElement * FilterDecision;
       // all events
       std::string FilterDirectory;
};

#endif
