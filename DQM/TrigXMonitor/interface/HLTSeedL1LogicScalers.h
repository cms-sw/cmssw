// -*- C++ -*-
//
// Package:    HLTSeedL1LogicScalers
// Class:      HLTSeedL1LogicScalers
// 
/**\class HLTSeedL1LogicScalers HLTSeedL1LogicScalers.cc DQM/HLTSeedL1LogicScalers/src/HLTSeedL1LogicScalers.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:   Vladimir  Rekovic
//         Created:  Tue Feb  9 15:15:20 CET 2010
// $Id: HLTSeedL1LogicScalers.h,v 1.4 2010/03/11 08:36:48 rekovic Exp $
//
//

#ifndef HLTSEEDSCALERS_H
#define HLTSEEDSCALERS_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"


#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class HLTSeedL1LogicScalers : public edm::EDAnalyzer {
   public:
      explicit HLTSeedL1LogicScalers(const edm::ParameterSet&);
      ~HLTSeedL1LogicScalers();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(const edm::Run& run, const edm::EventSetup& c) ;
      bool analyzeL1GtUtils(const edm::Event&, const edm::EventSetup&, const std::string &);
      bool analyzeL1GtRecord(const edm::Event&, const edm::EventSetup&, std::string);



      // ----------member data ---------------------------

      bool fL1BeforeMask;
      std::string fDQMFolder;
      std::string fProcessname;

      DQMStore * fDbe;

      L1GtUtils m_l1GtUtils;

      HLTConfigProvider fHLTConfig;
      /*
      edm::Handle<edm::TriggerResults> fTriggerResults;

      edm::InputTag fTriggerResultsLabel;
      edm::InputTag fL1GtLabel;
      */
      edm::InputTag fL1GtDaqReadoutRecordInputTag;
      edm::InputTag fL1GtRecordInputTag;


      std::vector<std::string> fMonitorPaths;
      std::vector<MonitorElement*> fMonitorPathsME;
      std::vector<std::pair<MonitorElement*, std::vector<std::string> > > fMapMEL1Algos;

};

#endif // HLTSEEDSCALERS_H
