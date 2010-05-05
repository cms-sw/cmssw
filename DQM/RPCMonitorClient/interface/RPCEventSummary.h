#ifndef RPCEventSummary_H
#define RPCEventSummary_H


/** \class RPCEventSummary
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2010/03/15 16:45:15 $
 *  $Revision: 1.15 $
 *  \author Anna Cimmino
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include <memory>
#include <string>


class DQMStore;
class RPCDetId;


class RPCEventSummary:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCEventSummary(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCEventSummary();

  /// BeginJob
  void beginJob();

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 

  
 private:
  
  void fillWithDefaultValue(float );
  float findRPCFED(const edm::EventSetup& );

  std::string eventInfoPath_, prefixDir_;

  bool enableReportSummary_;
  int prescaleFactor_, minimumEvents_;
  bool init_;
  DQMStore* dbe_;
  int event_;
  MonitorElement * RPCEvents_;
  int nLumiSegs_;
  std::string globalFolder_;
  float defaultValue_;

  std::pair<int, int> FEDRange_;
  int numberOfDisks_;  
  int NumberOfFeds_;
  enum RPCQualityFlags{DEAD = 6, PARTIALLY_DEAD=5};

};

#endif
