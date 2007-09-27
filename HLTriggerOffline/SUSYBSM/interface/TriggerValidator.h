#ifndef SusyBsmTriggerPerformance_TriggerValidator_TriggerValidator_h
#define SusyBsmTriggerPerformance_TriggerValidator_TriggerValidator_h

// -*- C++ -*-
//
// Package:    TriggerValidator
// Class:      TriggerValidator
// 
/**\class TriggerValidator TriggerValidator.cc SusyBsmTriggerPerformance/TriggerValidator/src/TriggerValidator.cc

 Description: Class to validate the Trigger Performance of the SUSYBSM group

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Massimiliano Chiorboli
//         Created:  Wed Aug 29 15:10:56 CEST 2007
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TH1.h"
#include "TList.h"

#include "HLTriggerOffline/SUSYBSM/interface/CutSelector.h"
#include "HLTriggerOffline/SUSYBSM/interface/PlotMaker.h"

class TriggerValidator : public edm::EDAnalyzer {
   public:
      explicit TriggerValidator(const edm::ParameterSet&);
      ~TriggerValidator();

      TFile* theHistoFile;

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------



      //CutSelector
      CutSelector* myCutSelector;
      PlotMaker* myPlotMaker;

      //Histo
      std::string HistoFileName;

      //Cut parameters decided by the user
      edm::ParameterSet userCut_params;
      edm::ParameterSet objectList;

      // name of each L1 algorithm
      std::vector<std::string> l1Names_;    
      // name of each hlt algorithm
      std::vector<std::string>  hlNames_;  

/*       //TList with the names of the bits */
/*       TList* lL1Names; */
/*       TList* lHLTNames; */


      //Counters for L1 and HLT
      std::vector<int> numTotL1BitsBeforeCuts;
      std::vector<int> numTotHltBitsBeforeCuts;
      std::vector<int> numTotL1BitsAfterCuts;
      std::vector<int> numTotHltBitsAfterCuts;
      
      //Histos for L1 and HLT bits
      TH1D* hL1BitsBeforeCuts;	
      TH1D* hHltBitsBeforeCuts;	
      TH1D* hL1BitsAfterCuts;	
      TH1D* hHltBitsAfterCuts;  

      TH1D* hL1PathsBeforeCuts;
      TH1D* hHltPathsBeforeCuts;
      TH1D* hL1PathsAfterCuts;
      TH1D* hHltPathsAfterCuts;
      
      TH2D* hL1HltMap;
      TH2D* hL1HltMapNorm;
      

      bool alreadyBooked;


};



#endif
