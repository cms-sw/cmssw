#ifndef MuonTrackResidualsTest_H
#define MuonTrackResidualsTest_H


/** \class MuonTrackResidualsTest
 * *
 *  DQMOffline Test Client
 *       check the residuals of the track parameters comparing STA/tracker only/global muons
 *
 *  $Date: 2009/12/22 17:43:26 $
 *  $Revision: 1.5 $
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
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>



class MuonTrackResidualsTest: public edm::EDAnalyzer{

public:

  /// Constructor
  MuonTrackResidualsTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonTrackResidualsTest();

protected:

  /// BeginJob
  void beginJob(void);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);


private:

  // counters
  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  // Switch for verbosity
  std::string metname;

  DQMStore* theDbe;
  edm::ParameterSet parameters;

  // source residuals histograms
  std::map< std::string, std::vector<std::string> > histoNames;

  // test histograms
  std::map< std::string, MonitorElement* > MeanHistos;
  std::map< std::string ,MonitorElement* > SigmaHistos;


};

#endif
