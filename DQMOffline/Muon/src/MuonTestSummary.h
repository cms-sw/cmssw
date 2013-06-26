#ifndef MuonTestSummary_H
#define MuonTestSummary_H


/** \class MuonTestSummary
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2013/05/30 22:14:15 $
 *  $Revision: 1.17 $
 *  \author  G. Mila - INFN Torino
 *  updates:  G. Hesketh - CERN
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
#include <string>

class MuonTestSummary: public edm::EDAnalyzer{

public:

  /// Constructor
  MuonTestSummary(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MuonTestSummary();

protected:

  /// BeginJob
  void beginJob(void);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c){}

  /// Histograms initialisation
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// test operations
  void doKinematicsTests(std::string muonType, int bin);
  void doResidualsTests(std::string type, std::string parameter, int bin);
  void doMuonIDTests();
  void doEnergyTests(std::string nameHisto, std::string muonType, int bin);
  void doMultiplicityTests();
  void ResidualCheck(std::string muType, const std::vector<std::string>& resHistos, int &numPlot, double &Mean, double &Mean_err, double &Sigma, double &Sigma_err);
  void GaussFit(std::string type, std::string parameter, MonitorElement *  Histo, float &mean, float &mean_err, float &sigma, float &sigma_err);


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
  double pullEtaSpread;
  double pullPhiSpread;
  double pullOneOvPSpread;
  double resChargeLimit_tkGlb;
  double resChargeLimit_glbSta;
  double resChargeLimit_tkSta;
  double numMatchedExpected_min;
  double numMatchedExpected_max;
  double matchesFractionDt_min;
  double matchesFractionDt_max;
  double matchesFractionCsc_min;
  double matchesFractionCsc_max;
  double resSegmTrack_rms_min;
  double resSegmTrack_rms_max;
  double resSegmTrack_mean_min;
  double resSegmTrack_mean_max;
  double sigmaResSegmTrackExp;
  double expPeakEcalS9_min;
  double expPeakEcalS9_max;
  double expPeakHadS9_min;
  double expPeakHadS9_max;
  double expMultiplicityGlb_min;
  double expMultiplicityTk_min;
  double expMultiplicitySta_min;
  double expMultiplicityGlb_max;
  double expMultiplicityTk_max;
  double expMultiplicitySta_max;

  // the report MEs
//------
  MonitorElement* KolmogorovTestSummaryMap;
  MonitorElement* chi2TestSummaryMap;
//-----
  MonitorElement* kinematicsSummaryMap;
  MonitorElement* residualsSummaryMap;
  MonitorElement* muonIdSummaryMap;
  MonitorElement* energySummaryMap;
  MonitorElement* multiplicitySummaryMap;
  MonitorElement* summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;
  MonitorElement* summaryCertification;
  MonitorElement*  summaryCertificationMap;
  std::vector<MonitorElement*>  theCertificationContents;

};

#endif
