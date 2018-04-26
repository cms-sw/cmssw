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
   verbose_(params.getUntrackedParameter<bool>("verbose", false)),
   gtUtil_(new l1t::L1TGlobalUtil(params, consumesCollector(), *this, params.getParameter<edm::InputTag>("l1tStage2uGtSource"), params.getParameter<edm::InputTag>("l1tStage2uGtSource"))),
   algoBitFirstBxInTrain_(-1),
   algoBitLastBxInTrain_(-1),
   algoBitIsoBx_(-1),
   algoNameFirstBxInTrain_(params.getUntrackedParameter<std::string>("firstBXInTrainAlgo", "")),
   algoNameLastBxInTrain_(params.getUntrackedParameter<std::string>("lastBXInTrainAlgo", "")),
   algoNameIsoBx_(params.getUntrackedParameter<std::string>("isoBXAlgo", "")),
   unprescaledAlgoShortList_(params.getUntrackedParameter<std::vector<std::string>> ("unprescaledAlgoShortList")),
   prescaledAlgoShortList_(params.getUntrackedParameter<std::vector<std::string>> ("prescaledAlgoShortList"))
{
  if (params.getUntrackedParameter<std::string>("useAlgoDecision").find("final") == 0) {
    useAlgoDecision_ = 2;
  } else if (params.getUntrackedParameter<std::string>("useAlgoDecision").find("intermediate") == 0) {
    useAlgoDecision_ = 1;
  } else {
    useAlgoDecision_ = 0;
  }
}

// Destructor
L1TStage2uGT::~L1TStage2uGT() {}

void L1TStage2uGT::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& evtSetup) {
   // Get the trigger menu information
   gtUtil_->retrieveL1Setup(evtSetup);
   // Get the algo bits needed for the timing histograms
   if (!gtUtil_->getAlgBitFromName(algoNameFirstBxInTrain_, algoBitFirstBxInTrain_)) {
      edm::LogWarning("L1TStage2uGT") << "Algo \"" << algoNameFirstBxInTrain_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
   }

   if (!gtUtil_->getAlgBitFromName(algoNameLastBxInTrain_, algoBitLastBxInTrain_)) {
      edm::LogWarning("L1TStage2uGT") << "Algo \"" << algoNameLastBxInTrain_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
   }

   if (!gtUtil_->getAlgBitFromName(algoNameIsoBx_, algoBitIsoBx_)) {
      edm::LogWarning("L1TStage2uGT") << "Algo \"" << algoNameIsoBx_ << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
   }

   int algoBitUnpre_=-1; 
   for(unsigned int i=0;i<unprescaledAlgoShortList_.size();i++){
     if (gtUtil_->getAlgBitFromName(unprescaledAlgoShortList_.at(i), algoBitUnpre_)) {
        unprescaledAlgoBitName_.emplace_back(unprescaledAlgoShortList_.at(i), algoBitUnpre_);
     }
     else {
       edm::LogWarning("L1TStage2uGT") << "Algo \"" << unprescaledAlgoShortList_.at(i) << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
     }
   }

   int algoBitPre_=-1; 
   for(unsigned int i=0;i<prescaledAlgoShortList_.size();i++){
     if ((gtUtil_->getAlgBitFromName(prescaledAlgoShortList_.at(i), algoBitPre_))) {
        prescaledAlgoBitName_.emplace_back(prescaledAlgoShortList_.at(i), algoBitPre_);
     }
     else {
       edm::LogWarning("L1TStage2uGT") << "Algo \"" << prescaledAlgoShortList_.at(i) << "\" not found in the trigger menu " << gtUtil_->gtTriggerMenuName() << ". Could not retrieve algo bit number.";
     }
   }

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
   const double preAlgs_d = static_cast<double>(prescaledAlgoBitName_.size());
   const double unpreAlgs_d = static_cast<double>(unprescaledAlgoBitName_.size());
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
   
   first_collision_minus2_run_ = ibooker.book2D("first_bunch_in_train_minus2", "uGT: Algorithm Trigger Bits (first bunch in train minus 2) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   first_collision_minus2_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   first_collision_minus2_run_->setAxisTitle("Algorithm Trigger Bits", 2);

   first_collision_minus1_run_ = ibooker.book2D("first_bunch_in_train_minus1", "uGT: Algorithm Trigger Bits (first bunch in train minus 1) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   first_collision_minus1_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   first_collision_minus1_run_->setAxisTitle("Algorithm Trigger Bits", 2);

   first_collision_run_ = ibooker.book2D("first_bunch_in_train", "uGT: Algorithm Trigger Bits (fisrt bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   first_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   first_collision_run_->setAxisTitle("Algorithm Trigger Bits (fisrt bunch in train)", 2);

   den_first_collision_minus2_run_ = ibooker.book2D("den_first_bunch_in_train_minus2", "uGT: Algorithm Trigger Bits (all entries for each trigget bit first bunch in train minus 2) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   den_first_collision_minus2_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   den_first_collision_minus2_run_->setAxisTitle("Algorithm Trigger Bits", 2);

   den_first_collision_minus1_run_ = ibooker.book2D("den_first_bunch_in_train_minus1", "uGT: Algorithm Trigger Bits (all entries for each trigget bit first bunch in train minus 1) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   den_first_collision_minus1_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   den_first_collision_minus1_run_->setAxisTitle("Algorithm Trigger Bits", 2);

   den_first_collision_run_ = ibooker.book2D("den_first_bunch_in_train", "uGT: Algorithm Trigger Bits (all entries for each trigget bit first bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   den_first_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   den_first_collision_run_->setAxisTitle("Algorithm Trigger Bits (fisrt bunch in train)", 2);

   last_collision_run_ = ibooker.book2D("last_bunch_in_train", "uGT: Algorithm Trigger Bits (last bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   last_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   last_collision_run_->setAxisTitle("Algorithm Trigger Bits (last bunch in train)", 2);

   den_last_collision_run_ = ibooker.book2D("den_last_bunch_in_train", "uGT: Algorithm Trigger Bits (all entries for each trigget bit last bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   den_last_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   den_last_collision_run_->setAxisTitle("Algorithm Trigger Bits (last bunch in train)", 2);

   isolated_collision_run_ = ibooker.book2D("isolated_bunch", "uGT: Algorithm Trigger Bits vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   isolated_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   isolated_collision_run_->setAxisTitle("Algorithm Trigger Bits", 2);

   den_isolated_collision_run_ = ibooker.book2D("den_isolated_bunch_in_train", "uGT: Algorithm Trigger Bits (all entries for each trigget bit isolated bunch in train) vs. BX Number In Event", 5, -2.5, 2.5, numAlgs, -0.5, numAlgs_d-0.5);
   den_isolated_collision_run_->setAxisTitle("Bunch Crossing Number In Event", 1);
   den_isolated_collision_run_->setAxisTitle("Algorithm Trigger Bits (isolated bunch in train)", 2);

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

  // Prescaled and Unprescaled Algo Trigger Bits
  // Two buckets before the bunch train
  prescaled_algo_two_bucket_before_train_ = ibooker.book2D("prescaled_algo_two_bucket_before_train_", "uGT: Prescaled Algorithm Trigger Bits  two buckets before train vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_two_bucket_before_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_two_bucket_before_train_->setAxisTitle("Algorithm Trigger Bits + Names ", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_two_bucket_before_train_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_two_bucket_before_train_den_ = ibooker.book2D("prescaled_algo_two_bucket_before_train_den_", "uGT: Prescaled Algorithm Trigger Bits Deno two buckets before train vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_two_bucket_before_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_two_bucket_before_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_two_bucket_before_train_den_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_two_bucket_before_train_ = ibooker.book2D("unprescaled_algo_two_bucket_before_train_", "uGT: Unprescaled Algorithm Trigger Bits  two buckets before train vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_two_bucket_before_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_two_bucket_before_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_two_bucket_before_train_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_two_bucket_before_train_den_ = ibooker.book2D("unprescaled_algo_two_bucket_before_train_den_", "uGT: Unprescaled Algorithm Trigger Bits Deno two buckets before train vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_two_bucket_before_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_two_bucket_before_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_two_bucket_before_train_den_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  // Bucket before the bunch train
  prescaled_algo_one_bucket_before_train_ = ibooker.book2D("prescaled_algo_one_bucket_before_train_", "uGT: Prescaled Algorithm Trigger Bits  one bucket before train vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_one_bucket_before_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_one_bucket_before_train_->setAxisTitle("Algorithm Trigger Bits + Names ", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_one_bucket_before_train_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_one_bucket_before_train_den_ = ibooker.book2D("prescaled_algo_one_bucket_before_train_den_", "uGT: Prescaled Algorithm Trigger Bits Deno one bucket before train vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_one_bucket_before_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_one_bucket_before_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_one_bucket_before_train_den_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_one_bucket_before_train_ = ibooker.book2D("unprescaled_algo_one_bucket_before_train_", "uGT: Unprescaled Algorithm Trigger Bits  one bucket before train vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_one_bucket_before_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_one_bucket_before_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_one_bucket_before_train_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_one_bucket_before_train_den_ = ibooker.book2D("unprescaled_algo_one_bucket_before_train_den_", "uGT: Unprescaled Algorithm Trigger Bits Deno one bucket before train vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_one_bucket_before_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_one_bucket_before_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_one_bucket_before_train_den_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  // Start of the bunch train
  prescaled_algo_first_collision_in_train_ = ibooker.book2D("prescaled_algo_first_collision_in_train_", "uGT: Prescaled Algorithm Trigger Bits  vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_first_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_first_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names ", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_first_collision_in_train_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_first_collision_in_train_den_ = ibooker.book2D("prescaled_algo_first_collision_in_train_den_", "uGT: Prescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_first_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_first_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_first_collision_in_train_den_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_first_collision_in_train_ = ibooker.book2D("unprescaled_algo_first_collision_in_train_", "uGT: Unprescaled Algorithm Trigger Bits  vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_first_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_first_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_first_collision_in_train_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_first_collision_in_train_den_ = ibooker.book2D("unprescaled_algo_first_collision_in_train_den_", "uGT: Unprescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_first_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_first_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_first_collision_in_train_den_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_isolated_collision_in_train_ = ibooker.book2D("prescaled_algo_isolated_collision_in_train_", "uGT: Prescaled Algorithm Trigger Bits vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_isolated_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_isolated_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_isolated_collision_in_train_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_isolated_collision_in_train_den_ = ibooker.book2D("prescaled_algo_isolated_collision_in_train_den_", "uGT: Prescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_isolated_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_isolated_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_isolated_collision_in_train_den_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_isolated_collision_in_train_ = ibooker.book2D("unprescaled_algo_isolated_collision_in_train_", "uGT: Unprescaled Algorithm Trigger Bits vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_isolated_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_isolated_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_isolated_collision_in_train_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_isolated_collision_in_train_den_ = ibooker.book2D("unprescaled_algo_isolated_collision_in_train_den_", "uGT: Unprescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_isolated_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_isolated_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_isolated_collision_in_train_den_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_last_collision_in_train_ = ibooker.book2D("prescaled_algo_last_collision_in_train_", "uGT: Prescaled Algorithm Trigger Bits vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_last_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_last_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_last_collision_in_train_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  prescaled_algo_last_collision_in_train_den_ = ibooker.book2D("prescaled_algo_last_collision_in_train_den_", "uGT: Prescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, prescaledAlgoBitName_.size(), -0.5, preAlgs_d-0.5);
  prescaled_algo_last_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  prescaled_algo_last_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) {
    prescaled_algo_last_collision_in_train_den_->setBinLabel(algo+1,prescaledAlgoBitName_.at(algo).first+"("+std::to_string(prescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_last_collision_in_train_ = ibooker.book2D("unprescaled_algo_last_collision_in_train_", "uGT: Unprescaled Algorithm Trigger Bits vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_last_collision_in_train_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_last_collision_in_train_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_last_collision_in_train_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

  unprescaled_algo_last_collision_in_train_den_ = ibooker.book2D("unprescaled_algo_last_collision_in_train_den_", "uGT: Unprescaled Algorithm Trigger Bits Deno vs. BX Number In Event", 5, -2.5, 2.5, unprescaledAlgoBitName_.size(), -0.5, unpreAlgs_d-0.5);
  unprescaled_algo_last_collision_in_train_den_->setAxisTitle("Bunch Crossing Number In Event", 1);
  unprescaled_algo_last_collision_in_train_den_->setAxisTitle("Algorithm Trigger Bits + Names", 2);
  for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
    unprescaled_algo_last_collision_in_train_den_->setBinLabel(algo+1,unprescaledAlgoBitName_.at(algo).first+"("+std::to_string(unprescaledAlgoBitName_.at(algo).second)+")",2);
  }

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

  // Find out in which BX the first collision in train, isolated bunch, and last collision in train have fired.
  // In case of pre firing it will be in BX 1 or BX 2 and this will determine the BX shift that
  // will be applied to the timing histogram later.
  int bxShiftFirst = -999;
  int bxShiftLast = -999;
  int bxShiftIso = -999;
  for (int bx = uGtAlgs->getFirstBX(); bx <= uGtAlgs->getLastBX(); ++bx) {
    for (GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(bx); itr != uGtAlgs->end(bx); ++itr) {
      // first bunch in train
      if (algoBitFirstBxInTrain_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitFirstBxInTrain_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitFirstBxInTrain_);
            break;
        }
        if (bit) {
          bxShiftFirst = bx;
        }
      }
      // last bunch in train
      if(algoBitLastBxInTrain_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitLastBxInTrain_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitLastBxInTrain_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitLastBxInTrain_);
            break;
        }
        if (bit) {
          bxShiftLast = bx;
        }
      }
      // isolated bunch
      if (algoBitIsoBx_ != -1) {
        bool bit = false;
        switch (useAlgoDecision_) {
          case 0:
            bit = itr->getAlgoDecisionInitial(algoBitIsoBx_);
            break;
          case 1:
            bit = itr->getAlgoDecisionInterm(algoBitIsoBx_);
            break;
          case 2:
            bit = itr->getAlgoDecisionFinal(algoBitIsoBx_);
            break;
        }
        if (bit) {
          bxShiftIso = bx;
        }
      }
    }
  }

  // fill the first bunch in train maps
  if (bxShiftFirst > -999) {
    auto minBx = std::max(uGtAlgs->getFirstBX(), uGtAlgs->getFirstBX() + bxShiftFirst);
    auto maxBx = std::min(uGtAlgs->getLastBX(), uGtAlgs->getLastBX() + bxShiftFirst);

    for (GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(bxShiftFirst); itr != uGtAlgs->end(bxShiftFirst); ++itr) {
      for (int ibx = minBx; ibx <= maxBx; ++ibx) {
        for (auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
          auto algoBits = itr2->getAlgoDecisionInitial(); 
          for (size_t algo = 0; algo < algoBits.size(); ++algo) { 
            if (algoBits.at(algo)) {
              first_collision_run_->Fill(ibx - bxShiftFirst, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                den_first_collision_run_->Fill(ibx2 - bxShiftFirst, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < prescaledAlgoBitName_.size(); ++algo) { 
            if (itr2->getAlgoDecisionInitial(prescaledAlgoBitName_.at(algo).second)) { 
              prescaled_algo_first_collision_in_train_->Fill(ibx - bxShiftFirst, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                prescaled_algo_first_collision_in_train_den_->Fill(ibx2 - bxShiftFirst, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < unprescaledAlgoBitName_.size(); ++algo) {
            if (itr2->getAlgoDecisionInitial(unprescaledAlgoBitName_.at(algo).second)) {
              unprescaled_algo_first_collision_in_train_->Fill(ibx - bxShiftFirst, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                unprescaled_algo_first_collision_in_train_den_->Fill(ibx2 - bxShiftFirst, algo);
              }
            }
          }
        }
      }
    }
  }

  // fill the last bunch in train maps
  if (bxShiftLast > -999) {
    auto minBx = std::max(uGtAlgs->getFirstBX(), uGtAlgs->getFirstBX() + bxShiftLast);
    auto maxBx = std::min(uGtAlgs->getLastBX(), uGtAlgs->getLastBX() + bxShiftLast);

    for (GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(bxShiftLast); itr != uGtAlgs->end(bxShiftLast); ++itr) {
      for (int ibx = minBx; ibx <= maxBx; ++ibx) {
        for (auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
          auto algoBits = itr2->getAlgoDecisionInitial(); 
          for (size_t algo = 0; algo < algoBits.size(); ++algo) { 
            if (algoBits.at(algo)) {
              last_collision_run_->Fill(ibx - bxShiftLast, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                den_last_collision_run_->Fill(ibx2 - bxShiftLast, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < prescaledAlgoBitName_.size(); ++algo) { 
            if (itr2->getAlgoDecisionInitial(prescaledAlgoBitName_.at(algo).second)) { 
              prescaled_algo_last_collision_in_train_->Fill(ibx - bxShiftLast, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                prescaled_algo_last_collision_in_train_den_->Fill(ibx2 - bxShiftLast, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < unprescaledAlgoBitName_.size(); ++algo) {
            if (itr2->getAlgoDecisionInitial(unprescaledAlgoBitName_.at(algo).second)) {
              unprescaled_algo_last_collision_in_train_->Fill(ibx - bxShiftLast, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                unprescaled_algo_last_collision_in_train_den_->Fill(ibx2 - bxShiftLast, algo);
              }
            }
          }
        }
      }
    }
  }

  // fill the isolated bunch maps
  if (bxShiftIso > -999) {
    auto minBx = std::max(uGtAlgs->getFirstBX(), uGtAlgs->getFirstBX() + bxShiftIso);
    auto maxBx = std::min(uGtAlgs->getLastBX(), uGtAlgs->getLastBX() + bxShiftIso);

    for (GlobalAlgBlkBxCollection::const_iterator itr = uGtAlgs->begin(bxShiftIso); itr != uGtAlgs->end(bxShiftIso); ++itr) {
      for (int ibx = minBx; ibx <= maxBx; ++ibx) {
        for (auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
          auto algoBits = itr2->getAlgoDecisionInitial(); 
          for (size_t algo = 0; algo < algoBits.size(); ++algo) { 
            if (algoBits.at(algo)) {
              isolated_collision_run_->Fill(ibx - bxShiftIso, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                den_isolated_collision_run_->Fill(ibx2 - bxShiftIso, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < prescaledAlgoBitName_.size(); ++algo) { 
            if (itr2->getAlgoDecisionInitial(prescaledAlgoBitName_.at(algo).second)) { 
              prescaled_algo_isolated_collision_in_train_->Fill(ibx - bxShiftIso, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                prescaled_algo_isolated_collision_in_train_den_->Fill(ibx2 - bxShiftIso, algo);
              }
            }
          }
          for (unsigned int algo = 0; algo < unprescaledAlgoBitName_.size(); ++algo) {
            if (itr2->getAlgoDecisionInitial(unprescaledAlgoBitName_.at(algo).second)) {
              unprescaled_algo_isolated_collision_in_train_->Fill(ibx - bxShiftIso, algo);
              for (int ibx2 = minBx; ibx2 <= maxBx; ++ibx2) {
                unprescaled_algo_isolated_collision_in_train_den_->Fill(ibx2 - bxShiftIso, algo);
              }
            }
          }
        }
      }
    }
  }

  // If algoBitFirstBxInTrain_ fired in L1A BX 2 something else must have prefired in the actual BX -2 before the first bunch crossing in the train 
  if (uGtAlgs->getLastBX() >= 2) {
    for(auto itr = uGtAlgs->begin(2); itr != uGtAlgs->end(2); ++itr) {
      if(algoBitFirstBxInTrain_ != -1 && itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_)) {
        for(int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
          for(auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
            auto algoBits = itr2->getAlgoDecisionInitial(); 
            for(size_t algo = 0; algo < algoBits.size(); ++algo) { 
              if(algoBits.at(algo)) {
                first_collision_minus2_run_->Fill(ibx, algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  den_first_collision_minus2_run_->Fill(ibx2, algo);
                }
              }
            }
            for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) { 
              if(itr2->getAlgoDecisionInitial(prescaledAlgoBitName_.at(algo).second)) { 
                prescaled_algo_two_bucket_before_train_->Fill(ibx, algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  prescaled_algo_two_bucket_before_train_den_->Fill(ibx2,algo);
                }
              }
            }
            for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
              if(itr2->getAlgoDecisionInitial(unprescaledAlgoBitName_.at(algo).second)) {
                unprescaled_algo_two_bucket_before_train_->Fill(ibx,algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  unprescaled_algo_two_bucket_before_train_den_->Fill(ibx2,algo);
                }
              }
            }
          }
        }
      }
    }
  }

  // If algoBitFirstBxInTrain_ fired in L1A BX 1 something else must have prefired in the actual BX -1 before the first bunch crossing in the train
  if (uGtAlgs->getLastBX() >= 1) {
    for(auto itr = uGtAlgs->begin(1); itr != uGtAlgs->end(1); ++itr) {
      if(algoBitFirstBxInTrain_ != -1 && itr->getAlgoDecisionInitial(algoBitFirstBxInTrain_)) {
        for(int ibx = uGtAlgs->getFirstBX(); ibx <= uGtAlgs->getLastBX(); ++ibx) {
          for(auto itr2 = uGtAlgs->begin(ibx); itr2 != uGtAlgs->end(ibx); ++itr2) {
            auto algoBits = itr2->getAlgoDecisionInitial(); 
            for(size_t algo = 0; algo < algoBits.size(); ++algo) { 
              if(algoBits.at(algo)) {
                first_collision_minus1_run_->Fill(ibx, algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  den_first_collision_minus1_run_->Fill(ibx2, algo);
                }
              }
            }
            for(unsigned int algo=0; algo<prescaledAlgoBitName_.size(); algo++) { 
              if(itr2->getAlgoDecisionInitial(prescaledAlgoBitName_.at(algo).second)) { 
                prescaled_algo_one_bucket_before_train_->Fill(ibx, algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  prescaled_algo_one_bucket_before_train_den_->Fill(ibx2,algo);
                }
              }
            }
            for(unsigned int algo=0; algo<unprescaledAlgoBitName_.size(); algo++) {
              if(itr2->getAlgoDecisionInitial(unprescaledAlgoBitName_.at(algo).second)) {
                unprescaled_algo_one_bucket_before_train_->Fill(ibx,algo);
                for(int ibx2 = uGtAlgs->getFirstBX(); ibx2 <= uGtAlgs->getLastBX(); ++ibx2) {
                  unprescaled_algo_one_bucket_before_train_den_->Fill(ibx2,algo);
                }
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
