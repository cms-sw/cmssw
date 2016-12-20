#ifndef L1TSTAGE2UGT_H
#define L1TSTAGE2UGT_H

/**
 * \class L1TStage2uGT
 *
 * Description: DQM for L1 Micro Global Trigger.
 *
 * \author Mateusz Zarucki 2016
 * \author J. Berryhill, I. Mikulec
 * \author Vasile Mihai Ghete - HEPHY Vienna
 *
 */

// System include files
#include <memory>
#include <vector>

// User include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/typedefs.h"

// L1 trigger include files
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

// DQM include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// Class declaration
//

class L1TStage2uGT: public DQMEDAnalyzer {

public:
   L1TStage2uGT(const edm::ParameterSet& ps); // constructor
   virtual ~L1TStage2uGT(); // destructor

protected:
   virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
   virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
   virtual void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override;
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&); // end section

private:
   
   // Input parameters
   edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtSource_; // input tag for L1 uGT DAQ readout record
  
   std::string monitorDir_; // histogram folder for L1 uGT plots
   
   bool verbose_; // verbosity switch
   
   // Booking of histograms for the module
   
   // Algorithm bits
   MonitorElement* algoBits_before_bxmask_;
   MonitorElement* algoBits_before_prescale_;
   MonitorElement* algoBits_after_prescale_;
  
   // Algorithm bits correlation
   MonitorElement* algoBits_before_bxmask_corr_;
   MonitorElement* algoBits_before_prescale_corr_;
   MonitorElement* algoBits_after_prescale_corr_;
 
   // Algorithm bits vs global BX number
   MonitorElement* algoBits_before_bxmask_bx_global_;
   MonitorElement* algoBits_before_prescale_bx_global_;
   MonitorElement* algoBits_after_prescale_bx_global_;
  
   // Algorithm bits vs BX number in event
   MonitorElement* algoBits_before_bxmask_bx_inEvt_;
   MonitorElement* algoBits_before_prescale_bx_inEvt_;
   MonitorElement* algoBits_after_prescale_bx_inEvt_;

   // Algorithm bits vs LS
   MonitorElement* algoBits_before_bxmask_lumi_;
   MonitorElement* algoBits_before_prescale_lumi_;
   MonitorElement* algoBits_after_prescale_lumi_;
 
   // Prescale factor index 
   MonitorElement* prescaleFactorSet_;

};

#endif
