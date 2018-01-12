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

void L1TStage2uGT::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& evtSetup, ugtdqm::Histograms& histograms) const {
   // empty 
}

void L1TStage2uGT::bookHistograms(DQMStore::ConcurrentBooker &booker, edm::Run const&, edm::EventSetup const& evtSetup, ugtdqm::Histograms& histograms) const {
   
   // Book histograms
   const int numLS = 2000;
   const double numLS_d = static_cast<double>(numLS);
   const int numAlgs = 512; // FIXME: Take number of algorithms from EventSetup
   const double numAlgs_d = static_cast<double>(numAlgs);
   const int numBx = 3564; 
   const double numBx_d = static_cast<double>(numBx);

   booker.setCurrentFolder(monitorDir_);
   
   // Algorithm bits 
   histograms.algoBits_before_bxmask = booker.book1D("algoBits_before_bxmask", "uGT: Algorithm Trigger Bits (before AlgoBX mask)", numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_bxmask.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 1);
   
   histograms.algoBits_before_prescale = booker.book1D("algoBits_before_prescale", "uGT: Algorithm Trigger Bits (before prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_prescale.setAxisTitle("Algorithm Trigger Bits (before prescale)", 1);
   
   histograms.algoBits_after_prescale = booker.book1D("algoBits_after_prescale", "uGT: Algorithm Trigger Bits (after prescale)", numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_after_prescale.setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
  
   // Algorithm bits correlation 
   histograms.algoBits_before_bxmask_corr = booker.book2D("algoBits_before_bxmask_corr","uGT: Algorithm Trigger Bit Correlation (before AlgoBX mask)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_bxmask_corr.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 1);
   histograms.algoBits_before_bxmask_corr.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   histograms.algoBits_before_prescale_corr = booker.book2D("algoBits_before_prescale_corr","uGT: Algorithm Trigger Bit Correlation (before prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_prescale_corr.setAxisTitle("Algorithm Trigger Bits (before prescale)", 1);
   histograms.algoBits_before_prescale_corr.setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   histograms.algoBits_after_prescale_corr = booker.book2D("algoBits_after_prescale_corr","uGT: Algorithm Trigger Bit Correlation (after prescale)", numAlgs, -0.5, numAlgs_d-0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_after_prescale_corr.setAxisTitle("Algorithm Trigger Bits (after prescale)", 1);
   histograms.algoBits_after_prescale_corr.setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs global BX number
   histograms.algoBits_before_bxmask_bx_global = booker.book2D("algoBits_before_bxmask_bx_global", "uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_bxmask_bx_global.setAxisTitle("Global Bunch Crossing Number", 1);
   histograms.algoBits_before_bxmask_bx_global.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   histograms.algoBits_before_prescale_bx_global = booker.book2D("algoBits_before_prescale_bx_global", "uGT: Algorithm Trigger Bits (before prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_prescale_bx_global.setAxisTitle("Global Bunch Crossing Number", 1);
   histograms.algoBits_before_prescale_bx_global.setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   histograms.first_collision_run = booker.book2D("first_bunch_in_train", "uGT: Algorithm Trigger Bits (fisrt bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.first_collision_run.setAxisTitle("Bunch Crossing Number In Event", 1);
   histograms.first_collision_run.setAxisTitle("Algorithm Trigger Bits (fisrt bunch in train)", 2);
   
   histograms.last_collision_run = booker.book2D("last_bunch_in_train", "uGT: Algorithm Trigger Bits (last bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.last_collision_run.setAxisTitle("Bunch Crossing Number In Event", 1);
   histograms.last_collision_run.setAxisTitle("Algorithm Trigger Bits (last bunch in train)", 2);

   histograms.isolated_collision_run = booker.book2D("isolated_bunch", "uGT: Algorithm Trigger Bits (Isolated bunch) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.isolated_collision_run.setAxisTitle("Bunch Crossing Number In Event", 1);
   histograms.isolated_collision_run.setAxisTitle("Algorithm Trigger Bits (Isolated bunch)", 2);

   histograms.algoBits_after_prescale_bx_global = booker.book2D("algoBits_after_prescale_bx_global", "uGT: Algorithm Trigger Bits (after prescale) vs. Global BX Number", numBx, 0.5, numBx_d + 0.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_after_prescale_bx_global.setAxisTitle("Global Bunch Crossing Number", 1);
   histograms.algoBits_after_prescale_bx_global.setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs BX number in event
   histograms.algoBits_before_bxmask_bx_inEvt = booker.book2D("algoBits_before_bxmask_bx_inEvt", "uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_bxmask_bx_inEvt.setAxisTitle("Bunch Crossing Number in Event", 1);
   histograms.algoBits_before_bxmask_bx_inEvt.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   histograms.algoBits_before_prescale_bx_inEvt = booker.book2D("algoBits_before_prescale_bx_inEvt", "uGT: Algorithm Trigger Bits (before prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_prescale_bx_inEvt.setAxisTitle("Bunch Crossing Number in Event", 1);
   histograms.algoBits_before_prescale_bx_inEvt.setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
   
   histograms.algoBits_after_prescale_bx_inEvt = booker.book2D("algoBits_after_prescale_bx_inEvt", "uGT: Algorithm Trigger Bits (after prescale) vs. BX Number in Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_after_prescale_bx_inEvt.setAxisTitle("Bunch Crossing Number in Event", 1);
   histograms.algoBits_after_prescale_bx_inEvt.setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);
  
   // Algorithm bits vs LS
   histograms.algoBits_before_bxmask_lumi = booker.book2D("algoBits_before_bxmask_lumi","uGT: Algorithm Trigger Bits (before AlgoBX mask) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_bxmask_lumi.setAxisTitle("Luminosity Segment", 1);
   histograms.algoBits_before_bxmask_lumi.setAxisTitle("Algorithm Trigger Bits (before AlgoBX mask)", 2);
   
   histograms.algoBits_before_prescale_lumi = booker.book2D("algoBits_before_prescale_lumi","uGT: Algorithm Trigger Bits (before prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_before_prescale_lumi.setAxisTitle("Luminosity Segment", 1);
   histograms.algoBits_before_prescale_lumi.setAxisTitle("Algorithm Trigger Bits (before prescale)", 2);
  
   histograms.algoBits_after_prescale_lumi = booker.book2D("algoBits_after_prescale_lumi","uGT: Algorithm Trigger Bits (after prescale) vs. LS", numLS, 0., numLS_d, numAlgs, -0.5, numAlgs_d-0.5);
   histograms.algoBits_after_prescale_lumi.setAxisTitle("Luminosity Segment", 1);
   histograms.algoBits_after_prescale_lumi.setAxisTitle("Algorithm Trigger Bits (after prescale)", 2);

   // Prescale factor index 
   histograms.prescaleFactorSet = booker.book2D("prescaleFactorSet", "uGT: Index of Prescale Factor Set vs. LS", numLS, 0., numLS_d, 25, 0., 25.);
   histograms.prescaleFactorSet.setAxisTitle("Luminosity Segment", 1);
   histograms.prescaleFactorSet.setAxisTitle("Prescale Factor Set Index", 2);
}

void L1TStage2uGT::dqmAnalyze(const edm::Event& evt, const edm::EventSetup& evtSetup, ugtdqm::Histograms const& histograms) const {
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
            histograms.prescaleFactorSet.fill(lumi, itr->getPreScColumn());
             
            // Fills algorithm bits histograms
            for(int algoBit = 0; algoBit < numAlgs; ++algoBit) {
              
               // Algorithm bits before AlgoBX mask 
               if(itr->getAlgoDecisionInitial(algoBit)) {
                  histograms.algoBits_before_bxmask.fill(algoBit);
                  histograms.algoBits_before_bxmask_lumi.fill(lumi, algoBit);
                  histograms.algoBits_before_bxmask_bx_inEvt.fill(ibx, algoBit); // FIXME: or itr->getbxInEventNr()/getbxNr()?
                  histograms.algoBits_before_bxmask_bx_global.fill(bx + ibx, algoBit);
 
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInitial(algoBit2)) {
                        histograms.algoBits_before_bxmask_corr.fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits before prescale 
               if(itr->getAlgoDecisionInterm(algoBit)) {
                  histograms.algoBits_before_prescale.fill(algoBit);
                  histograms.algoBits_before_prescale_lumi.fill(lumi, algoBit);
                  histograms.algoBits_before_prescale_bx_inEvt.fill(ibx, algoBit);
                  histograms.algoBits_before_prescale_bx_global.fill(bx + ibx, algoBit);

                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionInterm(algoBit2)) {
                        histograms.algoBits_before_prescale_corr.fill(algoBit, algoBit2);
                     }
                  }
               }  
               
               // Algorithm bits after prescale 
               if(itr->getAlgoDecisionFinal(algoBit)) {
                  histograms.algoBits_after_prescale.fill(algoBit);
                  histograms.algoBits_after_prescale_lumi.fill(lumi, algoBit);
                  histograms.algoBits_after_prescale_bx_inEvt.fill(ibx, algoBit);
                  histograms.algoBits_after_prescale_bx_global.fill(bx + ibx, algoBit);
                  
                  for(int algoBit2 = 0; algoBit2 < numAlgs; ++algoBit2) {
                     if(itr->getAlgoDecisionFinal(algoBit2)) {
                        histograms.algoBits_after_prescale_corr.fill(algoBit, algoBit2);
                     }
                  }
               }  
            }
         }
      }

  for(auto itr = uGtAlgs->begin(0); itr != uGtAlgs->end(0); ++itr) { 
//  This loop is only called once since the size of uGTAlgs seems to be always 1
     if(itr->getAlgoDecisionInitial(488)) {
//  Algo bit for the first bunch in train trigger (should be made configurable or, better, taken from conditions if possible)
//  The first BX in train trigger has fired. Now check all other triggers around this.
        for(int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
           for(auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
//  This loop is probably only called once since the size of uGtAlgs seems to be always 1
              auto algoBits = itr2->getAlgoDecisionInitial(); 
//  get a vector with all algo bits for this BX
              for(size_t algo = 0; algo < algoBits.size(); ++algo) { 
//  check all algos
                 if(algoBits.at(algo)) {
//  fill if the algo fired 
                    histograms.first_collision_run.fill(ibx, algo);
                 } //end of fired algo
              } // end of all algo trigger bits
           } // end of uGtAlgs
        } // end of BX
     } // selecting FirstCollisionInTrain
     if(itr->getAlgoDecisionInitial(488) && itr->getAlgoDecisionInitial(487)) {
        for(int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
           for(auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
              auto algoBits = itr2->getAlgoDecisionInitial();
              for(size_t algo = 0; algo < algoBits.size(); ++algo) {
                 if(algoBits.at(algo)) {
                    histograms.isolated_collision_run.fill(ibx, algo);
                 } //end of fired algo
              } // end of all algo trigger bits
           } // end of uGtAlgs
        } // end of BX
     } // selecting FirstCollisionInTrain && LastCollisionInTrain
     if(itr->getAlgoDecisionInitial(487)) {
        for(int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
           for(auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
              auto algoBits = itr2->getAlgoDecisionInitial();
              for(size_t algo = 0; algo < algoBits.size(); ++algo) {
                 if(algoBits.at(algo)) {
                    histograms.last_collision_run.fill(ibx, algo);
                 } //end of fired algo
              } // end of all algo trigger bits
           } // end of uGtAlgs
        } // end of BX
     } // selecting LastCollisionInTrain
  } // end of uGTAlgs = 1

 }
}
      
