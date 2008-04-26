#ifndef MuonTrackResidualsTest_H
#define MuonTrackResidualsTest_H


/** \class MuonTrackResidualsTest
 * *
 *  DQMOffline Test Client
 *       check the residuals of the track parameters comparing STA/tracker only/global muons
 *
 *  $Date: 2008/03/01 00:39:52 $
 *  $Revision: 1.10 $
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



class MuonTrackResidualsTest: public edm::EDAnalyzer{

public:

  /// Constructor
  MuonTrackResidualsTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonTrackResidualsTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(std::string type);

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

  // source residuals histograms
  std::map< std::string, std::vector<std::string> > histoNames;

  // test histograms
  std::map< std::string, MonitorElement* > MeanHistos;
  std::map< std::string ,MonitorElement* > SigmaHistos;


};

#endif
