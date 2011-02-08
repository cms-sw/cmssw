#ifndef RPCEventSummary_H
#define RPCEventSummary_H


/** \class RPCEventSummary
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2010/06/25 14:46:41 $
 *  $Revision: 1.19 $
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
  
   //End Run
   void endRun(const edm::Run& r, const edm::EventSetup& c);
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 

  
 private:
  void clientOperation();
  std::string eventInfoPath_, prefixDir_;

  bool tier0_;  
  bool enableReportSummary_;
  int prescaleFactor_, minimumEvents_;
  MonitorElement *  RPCEvents ;
  bool init_;
  DQMStore* dbe_;
  bool offlineDQM_;
  int lumiCounter_;
  std::string globalFolder_;
  
  int numberDisk_;
  bool   doEndcapCertification_;
  std::pair<int, int> FEDRange_;
  int NumberOfFeds_;

  enum RPCQualityFlags{DEAD = 6, PARTIALLY_DEAD=5};

};

#endif
