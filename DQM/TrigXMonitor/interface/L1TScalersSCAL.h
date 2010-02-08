#ifndef L1TScalersSCAL_H
#define L1TScalersSCAL_H

#include<vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class L1TScalersSCAL: public edm::EDAnalyzer
{
public:

  enum { N_LUMISECTION_TIME = 93 };

  /// Constructors
  L1TScalersSCAL(const edm::ParameterSet& ps);
      
  /// Destructor
  virtual ~L1TScalersSCAL();
            
  /// BeginJob
  void beginJob(const edm::EventSetup& c);
                
  /// Endjob
  void endJob(void);
                     
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
                         
  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);
                                                            
//   /// Begin LumiBlock
//   void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
//                             const edm::EventSetup& c) ;
                               
  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);
                                  
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
  edm::InputTag scalersSource_;
                                                                     
  std::string outputFile_;	//file name for ROOT ouput
  bool verbose_, monitorDaemon_;
  int nev_; // Number of events processed
  long reftime_,buffertime_;
  std::vector<double> algorithmRates_;
  std::vector<double> bufferAlgoRates_;
  std::vector<double> technicalRates_;
  std::vector<double> bufferTechRates_;                                 
  
  MonitorElement * orbitNum;
  MonitorElement * trigNum;
  MonitorElement * eventNum;
  MonitorElement * physTrig;
  MonitorElement * randTrig;
  MonitorElement * numberResets;
  MonitorElement * deadTime;
  MonitorElement * lostFinalTriggers;
  MonitorElement * algoRate[128];
  MonitorElement * techRate[64];


  MonitorElement * physRate;
  MonitorElement * randRate;
  MonitorElement * deadTimePercent;
  MonitorElement * lostPhysRate;
  MonitorElement * lostPhysRateBeamActive;
  MonitorElement * instTrigRate;
  MonitorElement * instEventRate;
  

  MonitorElement *  instLumi;
  MonitorElement *  instLumiErr; 
  MonitorElement *  instLumiQlty; 
  MonitorElement *  instEtLumi; 
  MonitorElement *  instEtLumiErr; 
  MonitorElement *  instEtLumiQlty; 
  MonitorElement *  sectionNum; 
  MonitorElement *  startOrbit; 
  MonitorElement *  numOrbits;   

 
  MonitorElement *  orbitNumL1A[4];
  MonitorElement *  bunchCrossingL1A[4];
  MonitorElement *  bunchCrossingCorr[3];
  MonitorElement *  bunchCrossingDiff[3];
  MonitorElement *  bunchCrossingDiff_small[3];


};

#endif // L1TScalersSCAL_H

