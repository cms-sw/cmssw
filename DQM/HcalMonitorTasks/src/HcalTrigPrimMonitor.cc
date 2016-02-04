#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

HcalTrigPrimMonitor::HcalTrigPrimMonitor (const edm::ParameterSet& ps) :
   dataLabel_(ps.getParameter<edm::InputTag>("dataLabel")),
   emulLabel_(ps.getParameter<edm::InputTag>("emulLabel")),
   ZSAlarmThreshold_(ps.getParameter< std::vector<int> >("ZSAlarmThreshold"))
{
   Online_                = ps.getUntrackedParameter<bool>("online",false);
   mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
   enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
   debug_                 = ps.getUntrackedParameter<int>("debug",false);
   prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
   if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
   subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","TrigPrimMonitor_Hcal"); 
   if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
   subdir_=prefixME_+subdir_;
   AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
   skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
   NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
   makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

}


HcalTrigPrimMonitor::~HcalTrigPrimMonitor () {
}


void
HcalTrigPrimMonitor::reset () {
}


void
HcalTrigPrimMonitor::setup() {
   HcalBaseDQMonitor::setup();
   
   if (dbe_ == 0)
      return;

   dbe_->setCurrentFolder(subdir_ + "TP Occupancy");
   TPOccupancyEta_ = dbe_->book1D("TPOccupancyVsEta", "TPOccupancyVsEta", 65, -32.5, 32.5);
   TPOccupancyPhi_ = dbe_->book1D("TPOccupancyVsPhi", "TPOccupancyVsPhi", 72, 0.5, 72.5);
   TPOccupancyPhiHFP_ = dbe_->book1D("TPOccupancyHFPVsPhi", "TPOccupancyHFPVsPhi", 72, 0.5, 72.5);
   TPOccupancyPhiHFM_ = dbe_->book1D("TPOccupancyHFMVsPhi", "TPOccupancyHFMVsPhi", 72, 0.5, 72.5);
   TPOccupancy_ = create_map(subdir_ + "TP Occupancy", "TPOccupancy");

   for (int isZS = 0; isZS <= 1; ++isZS) {

      std::string folder(subdir_);
      std::string zsname="_ZS";
      if (isZS == 0)
	{
	  folder += "noZS/";
	  zsname="_noZS";
	}

      std::string problem_folder(folder);
      problem_folder += "Problem TPs/";
      
      good_tps[isZS] = create_map(folder, "Good TPs"+zsname);
      bad_tps[isZS] = create_map(folder, "Bad TPs"+zsname);

      errorflag[isZS] = create_errorflag(folder, "Error Flag"+zsname);
      problem_map[isZS][kMismatchedEt] = create_map(problem_folder, "Mismatched Et"+zsname);
      problem_map[isZS][kMismatchedFG] = create_map(problem_folder, "Mismatched FG"+zsname);
      problem_map[isZS][kMissingData] = create_map(problem_folder, "Missing Data"+zsname);
      problem_map[isZS][kMissingEmul] = create_map(problem_folder, "Missing Emul"+zsname);

      for (int isHF = 0; isHF <= 1; ++isHF) {
         std::string subdet = (isHF == 0 ? "HBHE " : "HF ");
         tp_corr[isZS][isHF] = create_tp_correlation(folder, subdet + "TP Correlation"+zsname);
         fg_corr[isZS][isHF] = create_fg_correlation(folder, subdet + "FG Correlation"+zsname);

         problem_et[isZS][isHF][kMismatchedFG]
            = create_et_histogram(problem_folder + "TP Values/", subdet + "Mismatched FG"+zsname);

         problem_et[isZS][isHF][kMissingData]
            = create_et_histogram(problem_folder + "TP Values/", subdet + "Missing Data"+zsname);

         problem_et[isZS][isHF][kMissingEmul]
            = create_et_histogram(problem_folder + "TP Values/", subdet + "Missing Emul"+zsname);
      }//isHF
   }//isZS

   // Number of bad cells vs. luminosity block
   ProblemsVsLB = dbe_->bookProfile(
         "TotalBadTPs_HCAL_vs_LS",
         "Total Number of Bad HCAL TPs vs lumi section",
         NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);

   ProblemsVsLB_HB = dbe_->bookProfile(
         "TotalBadTPs_HB_vs_LS",
         "Total Number of Bad HB TPs vs lumi section",
         NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);

   ProblemsVsLB_HE = dbe_->bookProfile(
         "TotalBadTPs_HE_vs_LS",
         "Total Number of Bad HE TPs vs lumi section",
         NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);

   ProblemsVsLB_HF = dbe_->bookProfile(
         "TotalBadTPs_HF_vs_LS",
         "Total Number of Bad HF TPs vs lumi section",
         NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);

   // No TPs for HO, DO NOT fill this histogram
   ProblemsVsLB_HO = dbe_->bookProfile(
         "TotalBadTPs_HO_vs_LS",
         "Total Number of Bad HO TPs vs lumi section",
         NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);

   ProblemsVsLB->getTProfile()->SetMarkerStyle(20);
   ProblemsVsLB_HB->getTProfile()->SetMarkerStyle(20);
   ProblemsVsLB_HE->getTProfile()->SetMarkerStyle(20);
   ProblemsVsLB_HO->getTProfile()->SetMarkerStyle(20);
   ProblemsVsLB_HF->getTProfile()->SetMarkerStyle(20);
}

void HcalTrigPrimMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  HcalBaseDQMonitor::beginRun(run,c);
  if (mergeRuns_ && tevt_>0) return; // don't reset counters if merging runs
  if (tevt_==0) this->setup(); // create all histograms; not necessary if merging runs together
  if (mergeRuns_==false) this->reset(); // call reset at start of all runs
} // void HcalTrigPrimMonitor::beginRun()

void
HcalTrigPrimMonitor::analyze (edm::Event const &e, edm::EventSetup const &s) {
   if (!IsAllowedCalibType()) return;
   if (LumiInOrder(e.luminosityBlock())==false) return;

   edm::Handle<HcalTrigPrimDigiCollection> data_tp_col;
   if (!e.getByLabel(dataLabel_, data_tp_col)) {
      edm::LogWarning("HcalTrigPrimMonitor")<< dataLabel_<< " data TP not available";
      return;
   }

   edm::Handle<HcalTrigPrimDigiCollection> emul_tp_col;
   if (!e.getByLabel(emulLabel_, emul_tp_col)) {
      edm::LogWarning("HcalTrigPrimMonitor")<< emulLabel_<< " emul TP not available";
      return;
   }

   HcalBaseDQMonitor::analyze(e,s); // base class increments ievt_, etc. counters
   processEvent(data_tp_col, emul_tp_col);
}


void
HcalTrigPrimMonitor::processEvent (
      const edm::Handle <HcalTrigPrimDigiCollection>& data_tp_col,
      const edm::Handle <HcalTrigPrimDigiCollection>& emul_tp_col) {

   if(dbe_ == 0) 
      return;

   std::vector<int> errorflag_per_event[2][2];
   for (int isZS = 0; isZS <= 1; ++isZS) {
      for (int isHF = 0; isHF <= 1; ++isHF) {
         errorflag_per_event[isZS][isHF] = std::vector<int>(kNErrorFlag, 0);
      }//for isHF
   }//for isZS

   good_tps[0]->setBinContent(-1,-1,ievt_);
   bad_tps[0]->setBinContent(-1,-1,ievt_);
   good_tps[1]->setBinContent(-1,-1,ievt_);
   bad_tps[1]->setBinContent(-1,-1,ievt_);

   for (HcalTrigPrimDigiCollection::const_iterator data_tp = data_tp_col->begin();
                                                  data_tp != data_tp_col->end();
                                                  ++data_tp) {
      int ieta = data_tp->id().ieta();
      int iphi = data_tp->id().iphi();
      int isHF = data_tp->id().ietaAbs() >= 29 ? 1 : 0;


      //
      if (data_tp->SOI_compressedEt() > 0) {
         TPOccupancy_->Fill(ieta, iphi);
         TPOccupancyEta_->Fill(ieta);
         TPOccupancyPhi_->Fill(iphi);

         if (isHF) {
            if (ieta > 0) {
               TPOccupancyPhiHFP_->Fill(iphi);
            }
            else {
               TPOccupancyPhiHFM_->Fill(iphi);
            }
         }
      }

      //check missing from emulator
      HcalTrigPrimDigiCollection::const_iterator emul_tp = emul_tp_col->find(data_tp->id());
      if (emul_tp == emul_tp_col->end()) {
         bool pass_ZS = true;

         for (int i=0; i<data_tp->size(); ++i) {
            int dataEt(data_tp->sample(i).compressedEt());
            problem_et[0][isHF][kMissingEmul]->Fill(dataEt);

            if (dataEt > ZSAlarmThreshold_[abs(ieta)]) {
               problem_et[1][isHF][kMissingEmul]->Fill(dataEt);
               pass_ZS = false;
            }
         }//for tp sample

         problem_map[0][kMissingEmul]->Fill(ieta, iphi);
         ++errorflag_per_event[0][isHF][kMissingEmul];
         bad_tps[0]->Fill(ieta, iphi);

         if (!pass_ZS) {
            problem_map[1][kMissingEmul]->Fill(ieta, iphi);
            ++errorflag_per_event[1][isHF][kMissingEmul];
            bad_tps[1]->Fill(ieta, iphi);

            // counts per LS
            if (abs(ieta) <= 16)
               ++nBad_TP_per_LS_HB_;
            else if(abs(ieta) <= 28)
               ++nBad_TP_per_LS_HE_;
            else
               ++nBad_TP_per_LS_HF_;
         }
      } //emul tp not found
      else {
         bool mismatchedEt_noZS = false;
         bool mismatchedEt_ZS = false;
         bool mismatchedFG_noZS = false;
         bool mismatchedFG_ZS = false;

         for (int i=0; i<data_tp->size(); ++i) {
            int dataEt(data_tp->sample(i).compressedEt());
            int dataFG(data_tp->sample(i).fineGrain());
            int emulEt(emul_tp->sample(i).compressedEt());
            int emulFG(emul_tp->sample(i).fineGrain());

            int diff = abs(dataEt - emulEt);
            bool fill_corr_ZS = true;
            
            if (diff == 0) {
               if (dataFG != emulFG) {
                  mismatchedFG_noZS = true;
                  problem_et[0][isHF][kMismatchedFG]->Fill(dataEt);

                  // exclude mismatched FG when HF TP < ZS_AlarmThreshold
                  if (isHF == 1 && dataEt <= ZSAlarmThreshold_.at(abs(ieta))) {
                     // Do not fill ZS correlation plots.
                     fill_corr_ZS = false;
                  }
                  else {
                     mismatchedFG_ZS = true;
                     problem_et[1][isHF][kMismatchedFG]->Fill(dataEt);
                  }
               } // matched et but not fg
            }
            else {
               mismatchedEt_noZS = true;
               if (diff > ZSAlarmThreshold_.at(abs(ieta))) {
                  mismatchedEt_ZS = true;
                  fill_corr_ZS = false;
               }
            } // mismatche et

            // Correlation plots
            tp_corr[0][isHF]->Fill(dataEt, emulEt);
            fg_corr[0][isHF]->Fill(dataFG, emulFG);

            if (fill_corr_ZS) {
               tp_corr[1][isHF]->Fill(dataEt, emulEt);
               fg_corr[1][isHF]->Fill(dataFG, emulFG);
            }
         }//for tp sample

         // Fill Problem Map and error counts
         if (mismatchedEt_noZS) {
            problem_map[0][kMismatchedEt]->Fill(ieta, iphi);
            ++errorflag_per_event[0][isHF][kMismatchedEt];
         }
         if (mismatchedEt_ZS) {
            problem_map[1][kMismatchedEt]->Fill(ieta, iphi);
            ++errorflag_per_event[1][isHF][kMismatchedEt];
         }
         if (mismatchedFG_noZS) {
            problem_map[0][kMismatchedFG]->Fill(ieta, iphi);
            ++errorflag_per_event[0][isHF][kMismatchedFG];
         }
         if (mismatchedFG_ZS) {
            problem_map[1][kMismatchedFG]->Fill(ieta, iphi);
            ++errorflag_per_event[1][isHF][kMismatchedFG];
         }
         if (mismatchedEt_noZS || mismatchedFG_noZS)
            bad_tps[0]->Fill(ieta, iphi);
         else
            good_tps[0]->Fill(ieta, iphi);
         if (mismatchedEt_ZS || mismatchedFG_ZS) {
            bad_tps[1]->Fill(ieta, iphi);

            // counts per LS
            if (abs(ieta) <= 16)
               ++nBad_TP_per_LS_HB_;
            else if(abs(ieta) <= 28)
               ++nBad_TP_per_LS_HE_;
            else
               ++nBad_TP_per_LS_HF_;
         }
         else
            good_tps[1]->Fill(ieta, iphi);
      }//emul tp found
   }//for data_tp_col


   //check missing from data
   for (HcalTrigPrimDigiCollection::const_iterator emul_tp = emul_tp_col->begin();
                                                   emul_tp != emul_tp_col->end();
                                                   ++emul_tp) {
      int ieta(emul_tp->id().ieta());
      int iphi(emul_tp->id().iphi());
      int isHF = emul_tp->id().ietaAbs() >= 29 ? 1 : 0;

      HcalTrigPrimDigiCollection::const_iterator data_tp = data_tp_col->find(emul_tp->id());
      if (data_tp == data_tp_col->end()) {
         bool pass_ZS = true;

         for (int i=0; i<emul_tp->size(); ++i) {
            int emulEt(emul_tp->sample(i).compressedEt());
            problem_et[0][isHF][kMissingData]->Fill(emulEt);

            if (emulEt > ZSAlarmThreshold_[abs(ieta)]) {
               problem_et[1][isHF][kMissingData]->Fill(emulEt);
               pass_ZS = false;
            }
         }//for tp sample

         problem_map[0][kMissingData]->Fill(ieta, iphi);
         ++errorflag_per_event[0][isHF][kMissingData];
         bad_tps[0]->Fill(ieta, iphi);

         if (!pass_ZS) {
            problem_map[1][kMissingData]->Fill(ieta, iphi);
            ++errorflag_per_event[1][isHF][kMissingData];
            bad_tps[1]->Fill(ieta, iphi);

            // counts per LS
            if (abs(ieta) <= 16)
               ++nBad_TP_per_LS_HB_;
            else if(abs(ieta) <= 28)
               ++nBad_TP_per_LS_HE_;
            else
               ++nBad_TP_per_LS_HF_;
         }
      } //data tp not found
   } //for emul_tp_col

   // Fill error flag per event
   for (int isZS = 0; isZS <= 1; ++isZS) {
      for (int isHF = 0; isHF <= 1; ++isHF) {
         for (int i=0; i<kNErrorFlag; ++i) {
            if (errorflag_per_event[isZS][isHF][i] > 0)
               errorflag[isZS]->Fill(i, isHF);
         }//for i
      }//for isHF
   }//for isZS
}

void
HcalTrigPrimMonitor::cleanup() {
   if (!enableCleanup_) return;
   if (dbe_) {
      dbe_->setCurrentFolder(subdir_);
      dbe_->removeContents();

      dbe_->setCurrentFolder(subdir_ + "noZS/Problem TPs/TP Values");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_ + "noZS/Problem TPs");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_ + "noZS");

      dbe_->setCurrentFolder(subdir_ + "Problem TPs/TP Values");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_ + "Problem TPs");
      dbe_->removeContents();
   }
}

void HcalTrigPrimMonitor::endJob()
{
  if (enableCleanup_) cleanup(); 
}

void HcalTrigPrimMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) {
   if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
   HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
   ProblemsCurrentLB->Reset();
   // Rest counter
   nBad_TP_per_LS_HB_ = 0;
   nBad_TP_per_LS_HE_ = 0;
   nBad_TP_per_LS_HF_ = 0;
}

void HcalTrigPrimMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) {
   if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
   // Fill histograms for this LS
   ProblemsVsLB_HB->Fill(currentLS, nBad_TP_per_LS_HB_);
   ProblemsVsLB_HE->Fill(currentLS, nBad_TP_per_LS_HE_);
   ProblemsVsLB_HF->Fill(currentLS, nBad_TP_per_LS_HF_);
   ProblemsVsLB->Fill(currentLS, nBad_TP_per_LS_HB_ + nBad_TP_per_LS_HE_ + nBad_TP_per_LS_HF_);

   ProblemsCurrentLB->Fill(-1,-1,levt_);
   ProblemsCurrentLB->Fill(0,0, nBad_TP_per_LS_HB_);
   ProblemsCurrentLB->Fill(1,0, nBad_TP_per_LS_HE_);
   ProblemsCurrentLB->Fill(3,0, nBad_TP_per_LS_HF_);
}


MonitorElement*
HcalTrigPrimMonitor::create_summary(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   return dbe_->book2D(name, name, 65, -32.5, 32.5, 72, 0.5, 72.5);
}

MonitorElement*
HcalTrigPrimMonitor::create_errorflag(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   MonitorElement* element = dbe_->book2D(name, name, 4, 1, 5, 2, 0, 2);
   element->setBinLabel(1, "Mismatched E");
   element->setBinLabel(2, "Mismatched FG");
   element->setBinLabel(3, "Missing Data");
   element->setBinLabel(4, "Missing Emul");
   element->setBinLabel(1, "HBHE", 2);
   element->setBinLabel(2, "HF", 2);
   return element;
}

MonitorElement*
HcalTrigPrimMonitor::create_tp_correlation(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   MonitorElement* element = dbe_->book2D(name, name, 50, 0, 256, 50, 0, 256);
   element->setAxisTitle("data TP", 1);
   element->setAxisTitle("emul TP", 2);
   return element;
}

MonitorElement*
HcalTrigPrimMonitor::create_fg_correlation(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   MonitorElement* element = dbe_->book2D(name, name, 2, 0, 2, 2, 0, 2);
   element->setAxisTitle("data FG", 1);
   element->setAxisTitle("emul FG", 2);
   return element;
}

MonitorElement*
HcalTrigPrimMonitor::create_map(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   std::string title = name +";ieta;iphi";
   return dbe_->book2D(name, title, 65, -32.5, 32.5, 72, 0.5, 72.5);
}

MonitorElement*
HcalTrigPrimMonitor::create_et_histogram(const std::string& folder, const std::string& name) {
   edm::LogInfo("HcalTrigPrimMonitor") << "Creating MonitorElement " << name << " in folder " << folder << "\n";

   dbe_->setCurrentFolder(folder);
   return dbe_->book1D(name, name, 256, 0, 256);
}

DEFINE_FWK_MODULE (HcalTrigPrimMonitor);
