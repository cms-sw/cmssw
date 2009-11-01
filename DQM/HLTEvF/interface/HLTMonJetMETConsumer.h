#ifndef HLTMONJETMETCONSUMER_H
#define HLTMONJETMETCONSUMER_H
// -*- C++ -*-
//
// Package:    HLTMonJetMETConsumer
// Class:      HLTMonJetMETConsumer
// 
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed Jan 16 15:55:28 CET 2008
//
// Adapted from HLTMonElectronConsumer.h
// Adapted for JetMET by. J. Cammin
//
// $Id: HLTMonJetMETConsumer.h,v 1.1 2008/12/21 01:43:49 cammin Exp $
//
//


// system include files
#include <memory>
#include <unistd.h>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class HLTMonJetMETConsumer : public edm::EDAnalyzer {
   public:
      explicit HLTMonJetMETConsumer(const edm::ParameterSet&);
      ~HLTMonJetMETConsumer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data --------------------------- 

      DQMStore * dbe;

      static const int Nhist = 6; // max number of objects to monitor

      MonitorElement *injetmet_ref_Et[Nhist];
      MonitorElement *injetmet_ref_Eta[Nhist];
      MonitorElement *injetmet_ref_Phi[Nhist];

      MonitorElement *injetmet_probe_Et[Nhist];
      MonitorElement *injetmet_probe_Eta[Nhist];
      MonitorElement *injetmet_probe_Phi[Nhist];

      MonitorElement *outjetmet_Et[Nhist];
      MonitorElement *outjetmet_Eta[Nhist];
      MonitorElement *outjetmet_Phi[Nhist];
      
      MonitorElement *Eff_pt;
      MonitorElement *MEtemp;

      MonitorElement *outjetmet_Eff_Et[Nhist];
      MonitorElement *outjetmet_Eff_Eta[Nhist];
      MonitorElement *outjetmet_Eff_Phi[Nhist];

      std::string dirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      std::string outputFile_;

      int ievt;

      std::vector<std::string> theHLTRefLabels;  
      std::vector<std::string> theHLTProbeLabels;  
      int RefLabelSize, ProbeLabelSize;
      bool runConsumer;
      
};
#endif
