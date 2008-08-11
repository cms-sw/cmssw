#ifndef MuonRecoTest_H
#define MuonRecoTest_H


/** \class MuonRecoTest
 * *
 *  DQMOffline Test Client
 *       check the recostruction efficiency of Sta/Glb on eta, phi parameters
 *
 *  $Date: 2008/05/06 11:02:28 $
 *  $Revision: 1.2 $
 *  \author  G. Mila - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>



class MuonRecoTest: public edm::EDAnalyzer{

public:

  /// Constructor
  MuonRecoTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonRecoTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);


private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  // Switch for verbosity
  std::string metname;

  DQMStore* dbe;
  edm::ParameterSet parameters;

   //histo binning parameters
  int etaBin;
  double etaMin;
  double etaMax;

  int phiBin;
  double phiMin;
  double phiMax;

  // efficiency histograms
  MonitorElement* etaEfficiency;
  MonitorElement* phiEfficiency;

};

#endif
