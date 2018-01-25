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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/typedefs.h"

// L1 trigger include files
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

// DQM include files
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

namespace ugtdqm {
  struct Histograms {
    // Booking of histograms for the module

    // Algorithm bits
    ConcurrentMonitorElement algoBits_before_bxmask;
    ConcurrentMonitorElement algoBits_before_prescale;
    ConcurrentMonitorElement algoBits_after_prescale;

    // Algorithm bits correlation
    ConcurrentMonitorElement algoBits_before_bxmask_corr;
    ConcurrentMonitorElement algoBits_before_prescale_corr;
    ConcurrentMonitorElement algoBits_after_prescale_corr;

    // Algorithm bits vs global BX number
    ConcurrentMonitorElement algoBits_before_bxmask_bx_global;
    ConcurrentMonitorElement algoBits_before_prescale_bx_global;
    ConcurrentMonitorElement algoBits_after_prescale_bx_global;

    // Algorithm bits vs BX number in event
    ConcurrentMonitorElement algoBits_before_bxmask_bx_inEvt;
    ConcurrentMonitorElement algoBits_before_prescale_bx_inEvt;
    ConcurrentMonitorElement algoBits_after_prescale_bx_inEvt;

    // Algorithm bits vs LS
    ConcurrentMonitorElement algoBits_before_bxmask_lumi;
    ConcurrentMonitorElement algoBits_before_prescale_lumi;
    ConcurrentMonitorElement algoBits_after_prescale_lumi;

    // Prescale factor index
    ConcurrentMonitorElement prescaleFactorSet;

    // Pre- Post- firing timing dedicated plots
    ConcurrentMonitorElement first_collision_run;
    ConcurrentMonitorElement isolated_collision_run;
    ConcurrentMonitorElement last_collision_run;
  };
}

//
// Class declaration
//

class L1TStage2uGT: public DQMGlobalEDAnalyzer<ugtdqm::Histograms> {

public:
   L1TStage2uGT(const edm::ParameterSet& ps); // constructor
   ~L1TStage2uGT() override; // destructor

protected:
   void dqmBeginRun(const edm::Run&, const edm::EventSetup&, ugtdqm::Histograms&) const override;
   void bookHistograms(DQMStore::ConcurrentBooker &booker, edm::Run const&, edm::EventSetup const&, ugtdqm::Histograms&) const override;
   void dqmAnalyze(const edm::Event&, const edm::EventSetup&, ugtdqm::Histograms const&) const override;

private:
   
   // Input parameters
   edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtSource_; // input tag for L1 uGT DAQ readout record
  
   std::string monitorDir_; // histogram folder for L1 uGT plots

   
   bool verbose_; // verbosity switch

   // To get the algo bits corresponding to algo names
   std::shared_ptr<l1t::L1TGlobalUtil> gtUtil_;

   // For the timing histograms
   int algoBitFirstBxInTrain_;
   int algoBitLastBxInTrain_;
   const std::string algoNameFirstBxInTrain_;
   const std::string algoNameLastBxInTrain_;
   
};

#endif
