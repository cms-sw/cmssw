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
   monitorDir_(params.getUntrackedParameter<std::string> ("monitorDir", "")),
   verbose_(params.getUntrackedParameter<bool>("verbose", false))
{
   // empty
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

   ibooker.setCurrentFolder(monitorDir_);
   
   // Algorithm bits 
   algoBits_before_bxmask_ = ibooker.book1D("algoBits_before_bxmask", "uGT: Algorithm Trigger Bits (before AlgoBX mask)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_bxmask_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 1);
   
   algoBits_before_prescale_ = ibooker.book1D("algoBits_before_prescale", "uGT: Algorithm Trigger Bits (before prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_prescale_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 1);
   
   algoBits_after_prescale_ = ibooker.book1D("algoBits_after_prescale", "uGT: Algorithm Trigger Bits (after prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescale_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
  
   // Algorithm bits correlation 
   algoBits_before_bxmask_corr_ = ibooker.book2D("algoBits_before_bxmask_corr","uGT: Algorithm Trigger Bit Correlation (before AlgoBX mask)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_bxmask_corr_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 1);
   algoBits_before_bxmask_corr_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   algoBits_before_prescale_corr_ = ibooker.book2D("algoBits_before_prescale_corr","uGT: Algorithm Trigger Bit Correlation (before prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_prescale_corr_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 1);
   algoBits_before_prescale_corr_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   algoBits_after_prescale_corr_ = ibooker.book2D("algoBits_after_prescale_corr","uGT: Algorithm Trigger Bit Correlation (after prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescale_corr_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
   algoBits_after_prescale_corr_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs global BX number
   algoBits_before_bxmask_bx_global_ = ibooker.book2D("algoBits_before_bxmask_bx_global", "uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_bxmask_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_before_bxmask_bx_global_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   algoBits_before_prescale_bx_global_ = ibooker.book2D("algoBits_before_prescale_bx_global", "uGT: Algorithm Trigger Bits (before prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_prescale_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_before_prescale_bx_global_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   algoBits_after_prescale_bx_global_ = ibooker.book2D("algoBits_after_prescale_bx_global", "uGT: Algorithm Trigger Bits (after prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescale_bx_global_->setAxisTitle("Global Bunch Crossing Number", 1); 
   algoBits_after_prescale_bx_global_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs BX number in event
   algoBits_before_bxmask_bx_inEvt_ = ibooker.book2D("algoBits_before_bxmask_bx_inEvt", "uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_bxmask_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_before_bxmask_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   algoBits_before_prescale_bx_inEvt_ = ibooker.book2D("algoBits_before_prescale_bx_inEvt", "uGT: Algorithm Trigger Bits (before prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_prescale_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_before_prescale_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   algoBits_after_prescale_bx_inEvt_ = ibooker.book2D("algoBits_after_prescale_bx_inEvt", "uGT: Algorithm Trigger Bits (after prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescale_bx_inEvt_->setAxisTitle("Bunch Crossing Number in Event", 1);
   algoBits_after_prescale_bx_inEvt_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs LS
   algoBits_before_bxmask_lumi_ = ibooker.book2D("algoBits_before_bxmask_lumi","uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_bxmask_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_before_bxmask_lumi_->setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   algoBits_before_prescale_lumi_ = ibooker.book2D("algoBits_before_prescale_lumi","uGT: Algorithm Trigger Bits (before prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_before_prescale_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_before_prescale_lumi_->setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
  
   algoBits_after_prescale_lumi_ = ibooker.book2D("algoBits_after_prescale_lumi","uGT: Algorithm Trigger Bits (after prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   algoBits_after_prescale_lumi_->setAxisTitle("Luminosity Segment", 1);
   algoBits_after_prescale_lumi_->setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);

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
      for (int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
         for (auto itr = uGtAlgs->begin(ibx); itr != uGtAlgs->end(ibx); ++itr) { // FIXME: redundant loop?
            
            // Fills prescale factor set histogram
            prescaleFactorSet_->Fill(lumi, itr->getPreScColumn());
             
            // Fills algorithm bits histograms
            for(int algoBit = 0; algoBit < numAlgs; ++algoBit) {
              
               // Algorithm bits before AlgoBX mask 
               if(itr->getAlgoDecisionInitial(algoBit)) {
                  algoBits_before_bxmask_->Fill(algoBit);
                  algoBits_before_bxmask_lumi_->Fill(lumi, algoBit);
                  algoBits_before_bxmask_bx_inEvt_->Fill(ibx, algoBit); // FIXME: or itr->getbxInEventNr()/getbxNr()?
                  algoBits_before_bxmask_bx_global_->Fill(bx + ibx, algoBit);
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInitial(algoBit2)) {
                        algoBits_before_bxmask_corr_->Fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits before prescale 
               if(itr->getAlgoDecisionInterm(algoBit)) {
                  algoBits_before_prescale_->Fill(algoBit);
                  algoBits_before_prescale_lumi_->Fill(lumi, algoBit);
                  algoBits_before_prescale_bx_inEvt_->Fill(ibx, algoBit);
                  algoBits_before_prescale_bx_global_->Fill(bx + ibx, algoBit);
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInterm(algoBit2)) {
                        algoBits_before_prescale_corr_->Fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits after prescale 
               if(itr->getAlgoDecisionFinal(algoBit)) {
                  algoBits_after_prescale_->Fill(algoBit);
                  algoBits_after_prescale_lumi_->Fill(lumi, algoBit);
                  algoBits_after_prescale_bx_inEvt_->Fill(ibx, algoBit);
                  algoBits_after_prescale_bx_global_->Fill(bx + ibx, algoBit);
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionFinal(algoBit2)) {
                        algoBits_after_prescale_corr_->Fill(algoBit, algoBit2);
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
