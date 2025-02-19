#ifndef EBTriggerTowerTask_H
#define EBTriggerTowerTask_H

/*
 * \file EBTriggerTowerTask.h
 *
 * $Date: 2012/04/27 13:46:01 $
 * $Revision: 1.33 $
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <vector>

class MonitorElement;
class DQMStore;

class EBTriggerTowerTask : public edm::EDAnalyzer {

 public:

  /// Constructor
  EBTriggerTowerTask(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~EBTriggerTowerTask();

  /// number of trigger towers in eta
  static const int nTTEta; 

  /// number of trigger towers in phi
  static const int nTTPhi; 

  /// number of supermodules
  static const int nSM; 

 protected:

  /// Analyze
  void analyze(const edm::Event& e, 
	       const edm::EventSetup& c);

  /// BeginJob
  void beginJob(void);

  /// EndJob
  void endJob(void);

  /// BeginRun
  void beginRun(const edm::Run & r, const edm::EventSetup & c);

  /// EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);

  /// Reset
  void reset(void);

  /// Setup
  void setup(void);

  /// Cleanup
  void cleanup(void);

 private:
  
  /// 1D array
  typedef std::vector<MonitorElement*> array1;

  /// reserve an array to hold one histogram per supermodule
  void reserveArray( array1& array );

  /// process a collection of digis, either real or emulated
  void processDigis( const edm::Event& e, 
                     const edm::Handle<EcalTrigPrimDigiCollection>& digis, 
		     array1& meEtMap,
		     array1& meVeto,
		     const edm::Handle<EcalTrigPrimDigiCollection>& compDigis
		     = edm::Handle<EcalTrigPrimDigiCollection>(),
                     const edm::Handle<edm::TriggerResults>& hltResults
                     = edm::Handle<edm::TriggerResults>());


  /// book monitor elements for real, or emulated digis
  void setup( std::string const &nameext,
	      std::string const  &folder, 
	      bool emulated);
  

  /// local event counter
  int ievt_;

  /// Et vs ix vs iy, for each SM 
  array1 meEtMapReal_;

  /// fine grain veto vs iphi vs ieta, for each SM 
  array1 meVetoReal_;

  /// Emulated Et vs ix vs iy, for each SM 
  array1 meEtMapEmul_;

  /// Emulated fine grain veto vs iphi vs ieta, for each SM 
  array1 meVetoEmul_;

  /// error flag vs iphi vs ieta, for each SM
  /// the error flag is set to true in case of a discrepancy between 
  /// the emulator and the real data
  array1 meEmulError_;
  array1 meEmulMatch_;
  array1 meVetoEmulError_;

  /// init flag
  bool init_;

  /// DQM back-end interface
  DQMStore* dqmStore_;

  /// path to MEs
  std::string prefixME_;

  /// remove MEs
  bool enableCleanup_;

  /// merge MEs across runs
  bool mergeRuns_;

  /// to find the input collection of real digis 
  edm::InputTag realCollection_;

  /// to find the input collection of emulated digis
  edm::InputTag emulCollection_;

  /// to find the input collection of crystal digis
  edm::InputTag EBDigiCollection_;

  /// to find the input collection of HLT bits
  edm::InputTag HLTResultsCollection_;
  std::string HLTCaloHLTBit_;
  std::string HLTMuonHLTBit_;

  /// debug output root file. if empty, no output file created.
  std::string outputFile_;

  /// 1D emulator match 1D
  MonitorElement* meEmulMatchIndex1D_;
  MonitorElement* meEmulMatchMaxIndex1D_;

  /// ET spectrums for the whole EB
  MonitorElement* meEtSpectrumReal_;
  MonitorElement* meEtSpectrumEmul_;
  MonitorElement* meEtSpectrumEmulMax_;

  /// number and ET average of TP vs bx
  MonitorElement* meEtBxReal_;
  MonitorElement* meOccupancyBxReal_;

  /// TCC timing
  MonitorElement* meTCCTimingCalo_;
  MonitorElement* meTCCTimingMuon_;

};

#endif
