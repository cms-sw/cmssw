#ifndef SusyBsmTriggerPerformance_TriggerValidator_TriggerValidator_h
#define SusyBsmTriggerPerformance_TriggerValidator_TriggerValidator_h

// -*- C++ -*-
//
// Package:    TriggerValidator
// Class:      TriggerValidator
// 
/**\class TriggerValidator TriggerValidator.cc HLTriggerOffline/SUSYBSM/src/TriggerValidator.cc

 Description: Class to validate the Trigger Performance of the SUSYBSM group

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Massimiliano Chiorboli
//                   Maurizio Pierini
//                   Maria Spiropulu
//         Created:  Wed Aug 29 15:10:56 CEST 2007
// $Id: TriggerValidator.h,v 1.8 2009/04/07 16:36:46 chiorbo Exp $
//
//

// system include files
#include <memory>
#include <fstream>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTriggerOffline/SUSYBSM/interface/RecoSelector.h"
#include "HLTriggerOffline/SUSYBSM/interface/McSelector.h"

//To be included in a second stage
#include "HLTriggerOffline/SUSYBSM/interface/PlotMakerL1.h"
#include "HLTriggerOffline/SUSYBSM/interface/PlotMakerReco.h"
//#include "HLTriggerOffline/SUSYBSM/interface/TurnOnMaker.h"

//included for DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"




class TriggerValidator : public edm::EDAnalyzer {
   public:
      explicit TriggerValidator(const edm::ParameterSet&);
      ~TriggerValidator();

      TFile* theHistoFile;

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


      void beginRun(const edm::Run& run, const edm::EventSetup& c);
      void endRun(const edm::Run& run, const edm::EventSetup& c);

      
      std::map<int,std::string> l1NameMap;
      // ----------member data ---------------------------
      DQMStore * dbe_;
      std::string dirname_;
      HLTConfigProvider hltConfig_;
      // --- names of the folders in the dbe ---
      std::string triggerBitsDir ;
      std::string recoSelBitsDir ; 
      std::string mcSelBitsDir   ;      



      int nHltPaths;
      int nL1Bits;


      //RecoSelector
      RecoSelector* myRecoSelector;
      McSelector* myMcSelector;

      //For the moment I switch off the more complex plots
       PlotMakerL1* myPlotMakerL1; 
       PlotMakerReco* myPlotMakerReco; 
/*       TurnOnMaker* myTurnOnMaker; */

      //Histo
      std::string HistoFileName;
      std::string StatFileName;
      edm::InputTag l1Label;
      edm::InputTag hltLabel;

      //McFlag
      bool mcFlag;
      bool l1Flag;

      //Cut parameters decided by the user
      edm::ParameterSet userCut_params;
      edm::ParameterSet turnOn_params;
      edm::ParameterSet plotMakerL1Input;
      edm::ParameterSet plotMakerRecoInput;

      // name of each L1 algorithm
      std::vector<std::string> l1Names_;    
      // name of each hlt algorithm
      edm::TriggerNames triggerNames_;  // TriggerNames class
      std::vector<std::string>  hlNames_;  


      //Counters for L1 and HLT
      std::vector<int> numTotL1BitsBeforeCuts;
      std::vector<int> numTotHltBitsBeforeCuts;
      std::vector<int> numTotL1BitsAfterRecoCuts;
      std::vector<int> numTotHltBitsAfterRecoCuts;
      std::vector<int> numTotL1BitsAfterMcCuts;
      std::vector<int> numTotHltBitsAfterMcCuts;

      std::vector<double> effL1BeforeCuts;
      std::vector<double> effHltBeforeCuts;
      std::vector<double> effL1AfterRecoCuts;
      std::vector<double> effHltAfterRecoCuts;
      std::vector<double> effL1AfterMcCuts;
      std::vector<double> effHltAfterMcCuts;
      
      

      std::vector< std::vector<int> > vCorrL1;
      std::vector< std::vector<int> > vCorrHlt;
      std::vector< std::vector<double> > vCorrNormL1;
      std::vector< std::vector<double> > vCorrNormHlt;

      int nEvTot;
      int nEvRecoSelected;
      int nEvMcSelected;



      
      //Histos for L1 and HLT bits
      MonitorElement* hL1BitsBeforeCuts;	
      MonitorElement* hHltBitsBeforeCuts;	
      MonitorElement* hL1BitsAfterRecoCuts;	
      MonitorElement* hHltBitsAfterRecoCuts;  
      MonitorElement* hL1BitsAfterMcCuts;	
      MonitorElement* hHltBitsAfterMcCuts;  

      MonitorElement* hL1PathsBeforeCuts;
      MonitorElement* hHltPathsBeforeCuts;
      MonitorElement* hL1PathsAfterRecoCuts;
      MonitorElement* hHltPathsAfterRecoCuts;
      MonitorElement* hL1PathsAfterMcCuts;
      MonitorElement* hHltPathsAfterMcCuts;
        

      //if we want to keep these, probably thay have to move to the client      
/*       TH2D* hL1OverlapNormToTotal; */
/*       TH2D* hHltOverlapNormToTotal; */
/*       TH2D* hL1OverlapNormToLargestPath; */
/*       TH2D* hHltOverlapNormToLargestPath; */





      bool firstEvent;


};



#endif
