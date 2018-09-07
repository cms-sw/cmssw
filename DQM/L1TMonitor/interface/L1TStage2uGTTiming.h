#ifndef L1TSTAGE2UGTTIMING_H
#define L1TSTAGE2UGTTIMING_H

/**
 * \class L1TStage2uGTTiming
 *
 * Description: DQM for L1 Micro Global Trigger timing.
 *
 */

// System include files
#include <memory>
#include <vector>
#include <utility>
// User include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

// L1 trigger include files
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

// DQM include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// Class declaration
//

class L1TStage2uGTTiming: public DQMEDAnalyzer {

public:
  L1TStage2uGTTiming(const edm::ParameterSet& ps); // constructor
  ~L1TStage2uGTTiming() override; // destructor

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
   
  // Input parameters
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtSource_; // input tag for L1 uGT DAQ readout record

  std::string monitorDir_; // histogram folder for L1 uGT plots

  bool verbose_; // verbosity switch

  // To get the algo bits corresponding to algo names
  std::shared_ptr<l1t::L1TGlobalUtil> gtUtil_;

  int numAlgs_;

  // For the timing histograms
  int algoBitFirstBxInTrain_;
  int algoBitLastBxInTrain_;
  int algoBitIsoBx_;
  const std::string algoNameFirstBxInTrain_;
  const std::string algoNameLastBxInTrain_;
  const std::string algoNameIsoBx_;

  unsigned int useAlgoDecision_;
 
  std::vector<std::string> unprescaledAlgoShortList_;
  std::vector<std::string> prescaledAlgoShortList_; 

  std::vector<std::pair<std::string,int>> unprescaledAlgoBitName_;
  std::vector<std::pair<std::string,int>> prescaledAlgoBitName_; 

  // Pre- Post- firing timing dedicated plots
  MonitorElement* first_collision_in_train_minus2_;
  MonitorElement* first_collision_in_train_minus1_;
  MonitorElement* first_collision_in_train_;
  MonitorElement* last_collision_in_train_;
  MonitorElement* isolated_collision_;

  MonitorElement* den_first_collision_in_train_minus2_; 
  MonitorElement* den_first_collision_in_train_minus1_; 
  MonitorElement* den_first_collision_in_train_; 
  MonitorElement* den_last_collision_in_train_;
  MonitorElement* den_isolated_collision_;

  // Prescaled and unprescaled short list algo trigger bits
  MonitorElement* prescaled_algo_first_collision_in_train_;
  MonitorElement* unprescaled_algo_first_collision_in_train_;
  MonitorElement* prescaled_algo_isolated_collision_;
  MonitorElement* unprescaled_algo_isolated_collision_;
  MonitorElement* prescaled_algo_last_collision_in_train_;
  MonitorElement* unprescaled_algo_last_collision_in_train_;

  MonitorElement* den_prescaled_algo_first_collision_in_train_;
  MonitorElement* den_unprescaled_algo_first_collision_in_train_;
  MonitorElement* den_prescaled_algo_isolated_collision_;
  MonitorElement* den_unprescaled_algo_isolated_collision_;
  MonitorElement* den_prescaled_algo_last_collision_in_train_;
  MonitorElement* den_unprescaled_algo_last_collision_in_train_;
};

#endif
