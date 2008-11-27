#ifndef MuonTestSummary_H
#define MuonTestSummary_H


/** \class MuonTestSummary
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2008/11/26 14:26:10 $
 *  $Revision: 1.3 $
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
#include <string>

class MuonTestSummary: public edm::EDAnalyzer{

public:

  /// Constructor
  MuonTestSummary(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonTestSummary();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c){}

  /// Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// test operations
  void doKinematicsTests(std::string muonType, int bin, double weight);
  void doResidualsTests(std::string type, std::string parameter, int bin);
  void doMuonIDTests();
  void doEnergyTests(std::string nameHisto, std::string muonType, int bin);
  
private:

  DQMStore* dbe;
  // Switch for verbosity
  std::string metname;

  // test ranges
  double etaExpected;
  double phiExpected;
  double etaSpread;
  double phiSpread;
  double chi2Fraction;
  double chi2Spread;
  double resEtaSpread_tkGlb;
  double resEtaSpread_glbSta;
  double resPhiSpread_tkGlb;
  double resPhiSpread_glbSta;
  double numMatchedExpected;
  double sigmaResSegmTrackExp;

  // the report MEs
  MonitorElement* kinematicsSummaryMap;
  MonitorElement* residualsSummaryMap;
  MonitorElement* muonIdSummaryMap;
  MonitorElement* energySummaryMap;
  MonitorElement* summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;

};

#endif
