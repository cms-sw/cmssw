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
// $Id: TriggerValidator.h,v 1.2 2007/09/28 11:10:50 chiorbo Exp $
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

#include "TFile.h"
#include "TH1.h"
#include "TList.h"

#include "HLTriggerOffline/SUSYBSM/interface/RecoSelector.h"
#include "HLTriggerOffline/SUSYBSM/interface/McSelector.h"
#include "HLTriggerOffline/SUSYBSM/interface/PlotMaker.h"
#include "HLTriggerOffline/SUSYBSM/interface/TurnOnMaker.h"



class TriggerValidator : public edm::EDAnalyzer {
   public:
      explicit TriggerValidator(const edm::ParameterSet&);
      ~TriggerValidator();

      TFile* theHistoFile;

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      void writeHistos();
      virtual void endJob() ;

      std::map<int,std::string> l1NameMap;
      // ----------member data ---------------------------



      //RecoSelector
      RecoSelector* myRecoSelector;
      McSelector* myMcSelector;
      PlotMaker* myPlotMaker;
      TurnOnMaker* myTurnOnMaker;

      //Histo
      std::string HistoFileName;
      std::string StatFileName;
      edm::InputTag hltLabel;

      //McFlag
      bool mcFlag;

      //Cut parameters decided by the user
      edm::ParameterSet userCut_params;
      edm::ParameterSet turnOn_params;
      edm::ParameterSet objectList;

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
      TH1D* hL1BitsBeforeCuts;	
      TH1D* hHltBitsBeforeCuts;	
      TH1D* hL1BitsAfterRecoCuts;	
      TH1D* hHltBitsAfterRecoCuts;  
      TH1D* hL1BitsAfterMcCuts;	
      TH1D* hHltBitsAfterMcCuts;  

      TH1D* hL1PathsBeforeCuts;
      TH1D* hHltPathsBeforeCuts;
      TH1D* hL1PathsAfterRecoCuts;
      TH1D* hHltPathsAfterRecoCuts;
      TH1D* hL1PathsAfterMcCuts;
      TH1D* hHltPathsAfterMcCuts;
        
      TH2D* hL1HltMap;
      TH2D* hL1HltMapNorm;
      
      TH2D* hL1OverlapNormToTotal;
      TH2D* hHltOverlapNormToTotal;
      TH2D* hL1OverlapNormToLargestPath;
      TH2D* hHltOverlapNormToLargestPath;





      bool alreadyBooked;


};



#endif
