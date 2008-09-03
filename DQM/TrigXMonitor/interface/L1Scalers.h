// -*-c++-*-
#ifndef L1Scalers_H
#define L1Scalers_H
// $Id: L1Scalers.h,v 1.6 2008/09/02 02:35:32 wittich Exp $

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class L1Scalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  L1Scalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1Scalers() {};
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

   // Endjob
   void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
  int nev_; // Number of events processed
  
  bool verbose_;
  edm::InputTag l1GtDataSource_; // L1 Scalers
  std::string folderName_; // dqm folder name
  MonitorElement *l1scalers_;
  MonitorElement *l1techScalers_;
  MonitorElement *l1Correlations_;
  MonitorElement *bxNum_;

  // 2d versions
  MonitorElement *l1scalersBx_;
  MonitorElement *l1techScalersBx_;

  // Int
  MonitorElement *nLumiBlock_;


};

#endif // L1Scalers_H
