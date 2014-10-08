#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/** \class HcalTrigPrimMonitor
  *  
  * \author W. Fisher - FNAL
  */

//class HcalTrigPrimMonitor: public HcalBaseMonitor {
class HcalTrigPrimMonitor: public HcalBaseDQMonitor {
   public:
      HcalTrigPrimMonitor(const edm::ParameterSet& ps);
      ~HcalTrigPrimMonitor(); 
  
      void setup(DQMStore::IBooker &);
      void analyze(const edm::Event& e, const edm::EventSetup& c);
      void processEvent(const edm::Handle<HcalTrigPrimDigiCollection>& data_tp_col,
                        const edm::Handle<HcalTrigPrimDigiCollection>& emul_tp_col);
      void reset();
      void bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c);
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) ;
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
      void endJob();

   private:
      edm::InputTag dataLabel_;
      edm::InputTag emulLabel_;

      edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_data_;
      edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_emu_;

      std::vector<int> ZSBadTPThreshold_;
      std::vector<int> ZSAlarmThreshold_;

      MonitorElement* create_summary(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);
      MonitorElement* create_errorflag(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);
      MonitorElement* create_tp_correlation(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);
      MonitorElement* create_fg_correlation(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);
      MonitorElement* create_map(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);
      MonitorElement* create_et_histogram(DQMStore::IBooker &ib, const std::string& folder, const std::string& name);

      enum ErrorFlag{
         kZeroTP=-1,
         kMatched = 0,
         kMismatchedEt = 1,
         kMismatchedFG = 2,
         kMissingData = 3,
         kMissingEmul = 4,
         kNErrorFlag = 5,
         kUnknown = kNErrorFlag
      };

      // Index: [isZS]
      MonitorElement* good_tps[2];
      MonitorElement* bad_tps[2];
      MonitorElement* errorflag[2];
      std::map<ErrorFlag, MonitorElement*> problem_map[2];

      // Index: [isZS], for OOT TPs
      MonitorElement* good_tps_oot[2];
      MonitorElement* bad_tps_oot[2];
      MonitorElement* errorflag_oot[2];
      std::map<ErrorFlag, MonitorElement*> problem_map_oot[2];

      // Index: [isZS][isHF]
      MonitorElement* tp_corr[2][2];
      MonitorElement* fg_corr[2][2];
      std::map<ErrorFlag, MonitorElement*> problem_et[2][2];

      // Index: [isZS][isHF], for OOT TPs
      MonitorElement* tp_corr_oot[2][2];
      MonitorElement* fg_corr_oot[2][2];
      std::map<ErrorFlag, MonitorElement*> problem_et_oot[2][2];

      MonitorElement* TPOccupancy_;
      MonitorElement* TPOccupancyEta_;
      MonitorElement* TPOccupancyPhi_;
      MonitorElement* TPOccupancyPhiHFP_;
      MonitorElement* TPOccupancyPhiHFM_;

      int nBad_TP_per_LS_HB_;
      int nBad_TP_per_LS_HE_;
      int nBad_TP_per_LS_HF_;
};
#endif
