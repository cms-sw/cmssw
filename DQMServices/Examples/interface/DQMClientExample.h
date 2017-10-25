#ifndef DQMClientExample_H
#define DQMClientExample_H

/** \class DQMClientExample
 * *
 *  DQM Test Client
 *
 *  \author  M. Zanetti CERN
 *   
 */


/* Jan 17, 2009: the code has been modified significantly
 * to steer the client operations  
 * Author: D.Volyanskyy
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>


class DQMClientExample: public edm::EDAnalyzer {

public:

  ////---- constructor
  DQMClientExample(const edm::ParameterSet& ps);
  
  ////---- destructor
  ~DQMClientExample() override;

protected:

  ////---- beginJob
  void beginJob() override;

  ////---- beginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  ////---- analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override ;

  ////---- performClient
  void performClient();

  ////---- beginLuminosityBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) override ;

  ////--- endLuminosityBlock 
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                         const edm::EventSetup& c) override;

  ////---- endRun
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

  ////---- endJob
  void endJob() override;

private:
  ////---- initialize
  void initialize();
  
  edm::ParameterSet parameters_;

  DQMStore* dbe_;  
  std::string monitorName_;
  std::string QTestName_;
  int counterClientOperation ; //-- counter on Client Operations
  int counterEvt_;     //-- event counter
  int counterLS_;     //-- LS counter
  int prescaleEvt_;    //-- prescale on number of events
  int prescaleLS_;    //-- prescale on number of lumisections
  bool clientOnEachEvent;
  bool clientOnEndLumi;
  bool clientOnEndRun;
  bool clientOnEndJob;

  // -------- member data --------
  MonitorElement * clientHisto;

};

#endif


