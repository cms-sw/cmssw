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

#include "DQM/L1TMonitor/interface/L1TStage2uGT.h"

// Constructor
L1TStage2uGT::L1TStage2uGT(const edm::ParameterSet& params):
   l1tStage2uGtSource_(consumes<GlobalAlgBlkBxCollection>(params.getParameter<edm::InputTag>("l1tStage2uGtSource"))),
   verbose_(params.getUntrackedParameter<bool>("verbose", false))
{
   histFolder_ = params.getUntrackedParameter<std::string> ("HistFolder", "L1T2016/L1TStage2uGT");
}

// Destructor
L1TStage2uGT::~L1TStage2uGT() {
   // empty
}

void L1TStage2uGT::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& evtSetup) {
   // empty 
}

void L1TStage2uGT::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& evtSetup) { 
   // empty
}

void L1TStage2uGT::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const& evtSetup) {
   
   // Book histograms
   const int numLS = 2000;
   const double numLS_d = static_cast<double>(numLS);
   const int numAlgs = 512; // FIXME: Take number of algorithms from EventSetup
   const double numAlgs_d = static_cast<double>(numAlgs);
   const int numBx = 3564; 
   const double numBx_d = static_cast<double>(numBx);

   ibooker.setCurrentFolder(histFolder_);
   
   // Algorithm bits 
   algoBits_after_bxomask_ = ibooker.book1D("algoBits_after_bxomask", "uGT: Algorithm Trigger Bits (after BX mask, before prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_bxomask_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 1);
   
   algoBits_after_prescaler_ = ibooker.book1D("algoBits_after_prescaler", "uGT: Algorithm Trigger Bits (after prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescaler_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
   
   algoBits_after_mask_ = ibooker.book1D("algoBits_after_mask", "uGT: Algorithm Trigger Bits (after mask)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_mask_->setAxisTitle("Algorithm Trigger Bits (after mask)", 1);
  
   // Algorithm bits correlation 
   algoBits_after_bxomask_corr_ = ibooker.book2D("algoBits_after_bxomask_corr","uGT: Algorithm Trigger Bit Correlation (after BX mask, before prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_bxomask_corr_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 1);
   algoBits_after_bxomask_corr_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 2);
   
   algoBits_after_prescaler_corr_ = ibooker.book2D("algoBits_after_prescaler_corr","uGT: Algorithm Trigger Bit Correlation (after prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescaler_corr_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
   algoBits_after_prescaler_corr_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
   
   algoBits_after_mask_corr_ = ibooker.book2D("algoBits_after_mask_corr","uGT: Algorithm Trigger Bit Correlation (after mask)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_mask_corr_->setAxisTitle("Algorithm Trigger Bits (after mask)", 1);
   algoBits_after_mask_corr_->setAxisTitle("Algorithm Trigger Bits (after mask)", 2);
  
   // Algorithm bits vs global BX number
   algoBits_after_bxomask_bx_global_ = ibooker.book2D("algoBits_after_bxomask_bx_global", "uGT: Algorithm Trigger Bits (after BX mask, before prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_bxomask_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_after_bxomask_bx_global_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 2);
   
   algoBits_after_prescaler_bx_global_ = ibooker.book2D("algoBits_after_prescaler_bx_global", "uGT: Algorithm Trigger Bits (after prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescaler_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_after_prescaler_bx_global_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
   
   algoBits_after_mask_bx_global_ = ibooker.book2D("algoBits_after_mask_bx_global", "uGT: Algorithm Trigger Bits (after mask) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_mask_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_after_mask_bx_global_->setAxisTitle("Algorithm Trigger Bits (after mask)", 2);
  
   // Algorithm bits vs BX number in event
   algoBits_after_bxomask_bx_inEvt_ = ibooker.book2D("algoBits_after_bxomask_bx_inEvt", "uGT: Algorithm Trigger Bits (after BX mask, before prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_bxomask_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_after_bxomask_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 2);
   
   algoBits_after_prescaler_bx_inEvt_ = ibooker.book2D("algoBits_after_prescaler_bx_inEvt", "uGT: Algorithm Trigger Bits (after prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescaler_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_after_prescaler_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
   
   algoBits_after_mask_bx_inEvt_ = ibooker.book2D("algoBits_after_mask_bx_inEvt", "uGT: Algorithm Trigger Bits (after mask) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_mask_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_after_mask_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (after mask)", 2);
  
   // Algorithm bits vs LS
   algoBits_after_bxomask_lumi_ = ibooker.book2D("algoBits_after_bxomask_lumi","uGT: Algorithm Trigger Bits (after BX mask, before prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_bxomask_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_after_bxomask_lumi_->setAxisTitle("Algorithm Trigger Bits (after BX mask, before prescale)", 2);
   
   algoBits_after_prescaler_lumi_ = ibooker.book2D("algoBits_after_prescaler_lumi","uGT: Algorithm Trigger Bits (after prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescaler_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_after_prescaler_lumi_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   algoBits_after_mask_lumi_ = ibooker.book2D("algoBits_after_mask_lumi","uGT: Algorithm Trigger Bits (after mask) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_mask_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_after_mask_lumi_->setAxisTitle("Algorithm Trigger Bits (after mask)", 2);

   // Prescale factor index 
   prescaleFactorSet_ = ibooker.book2D("prescaleFactorSet", "uGT: Index of Prescale Factor Set vs. LS", numLS, 0., numLS_d, 25, 0., 25.);
   prescaleFactorSet_->setAxisTitle("Luminosity Segment", 1);
   prescaleFactorSet_->setAxisTitle("Prescale Factor Set Index", 2);
}

void L1TStage2uGT::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
   // FIXME: Remove duplicate definition of numAlgs 
   const int numAlgs = 512;
  
   if (verbose_) {
      edm::LogInfo("L1TStage2uGT") << "L1TStage2uGT DQM: Analyzing.." << std::endl;
   }
   
   // Get standard event parameters 
   int lumi = evt.luminosityBlock();
   int bx = evt.bunchCrossing();
      
   // Open uGT readout record
   edm::Handle<GlobalAlgBlkBxCollection> uGtAlgs;
   evt.getByToken(l1tStage2uGtSource_, uGtAlgs);
   
   if (!uGtAlgs.isValid()) {
      edm::LogInfo("L1TStage2uGT") << "Cannot find uGT readout record.";
      return;
   }
   
   // Get uGT algo bit statistics
   else {
      //algoBits_->Fill(-1.); // fill underflow to normalize // FIXME: needed? 
      for (int ibx=uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
         for (auto itr = uGtAlgs->begin(ibx); itr != uGtAlgs->end(ibx); ++itr) { // FIXME: redundant loop over 1-dim vector?
            
            // Fills prescale factor set histogram
            prescaleFactorSet_->Fill(lumi, itr->getPreScColumn());
             
            // Fills algorithm bits histograms
            for(int algoBit = 0; algoBit < numAlgs; ++algoBit) {
              
               // Algorithm bits after BX mask, before prescale 
               if(itr->getAlgoDecisionInitial(algoBit)) {
                  algoBits_after_bxomask_->Fill(algoBit);
                  algoBits_after_bxomask_lumi_->Fill(lumi, algoBit);
                  algoBits_after_bxomask_bx_global_->Fill(bx, algoBit);
                  algoBits_after_bxomask_bx_inEvt_->Fill(ibx, algoBit); // FIXME: or itr->getbxInEventNr()/getbxNr()?
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInitial(algoBit2)) {
                        algoBits_after_bxomask_corr_->Fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits after prescale 
               if(itr->getAlgoDecisionInterm(algoBit)) {
                  algoBits_after_prescaler_->Fill(algoBit);
                  algoBits_after_prescaler_lumi_->Fill(lumi, algoBit);
                  algoBits_after_prescaler_bx_global_->Fill(bx, algoBit);
                  algoBits_after_prescaler_bx_inEvt_->Fill(ibx, algoBit); // FIXME: or itr->getbxInEventNr()/getbxNr()?
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInterm(algoBit2)) {
                        algoBits_after_prescaler_corr_->Fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits after mask 
               if(itr->getAlgoDecisionFinal(algoBit)) {
                  algoBits_after_mask_->Fill(algoBit);
                  algoBits_after_mask_lumi_->Fill(lumi, algoBit);
                  algoBits_after_mask_bx_global_->Fill(bx, algoBit);
                  algoBits_after_mask_bx_inEvt_->Fill(ibx, algoBit); // FIXME: or itr->getbxInEventNr()/getbxNr()?
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionFinal(algoBit2)) {
                        algoBits_after_mask_corr_->Fill(algoBit, algoBit2);
                     }
                  }
               }  
            }
         }
      }
   }
}

// End section
void L1TStage2uGT::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& evtSetup) {
   // empty
}
