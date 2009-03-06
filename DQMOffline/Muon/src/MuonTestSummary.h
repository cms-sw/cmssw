#ifndef MuonTestSummary_H
#define MuonTestSummary_H


/** \class MuonTestSummary
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2008/12/17 13:57:52 $
 *  $Revision: 1.7 $
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

  /// Histograms initialisation
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// test operations
  void doKinematicsTests(std::string muonType, int bin);
  void doResidualsTests(std::string type, std::string parameter, int bin);
  void doMuonIDTests();
  void doEnergyTests(std::string nameHisto, std::string muonType, int bin);
  void doMolteplicityTests();
  
private:

  DQMStore* dbe;
  // Switch for verbosity
  std::string metname;

  // test ranges
  double etaExpected;
  double phiExpected;
  double chi2Fraction;
  double chi2Spread;
  double resEtaSpread_tkGlb;
  double resEtaSpread_glbSta;
  double resPhiSpread_tkGlb;
  double resPhiSpread_glbSta;
  double resOneOvPSpread_tkGlb;
  double resOneOvPSpread_glbSta;
  double resChargeLimit_tkGlb;
  double resChargeLimit_glbSta;
  double resChargeLimit_tkSta;
  double numMatchedExpected;
  double sigmaResSegmTrackExp;
  double expMolteplicityGlb;
  double expMolteplicityTk;
  double expMolteplicitySta;

  // the report MEs
  MonitorElement* kinematicsSummaryMap;
  MonitorElement* residualsSummaryMap;
  MonitorElement* muonIdSummaryMap;
  MonitorElement* energySummaryMap;
  MonitorElement* molteplicitySummaryMap;
  MonitorElement* summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;

};

#endif
