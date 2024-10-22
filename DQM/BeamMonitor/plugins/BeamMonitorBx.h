#ifndef BeamMonitorBx_H
#define BeamMonitorBx_H

/** \class BeamMonitorBx
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *   
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include <fstream>

//
// class declaration
//

class BeamMonitorBx
    : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  BeamMonitorBx(const edm::ParameterSet&);
  ~BeamMonitorBx() override;

  typedef int BxNum;
  typedef std::map<BxNum, reco::BeamSpot> BeamSpotMapBx;

protected:
  // BeginJob
  void beginJob() override;

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;
  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  void FitAndFill(const edm::LuminosityBlock& lumiSeg, int&, int&, int&);
  void BookTables(int, std::map<std::string, std::string>&, std::string);
  void BookTrendHistos(bool, int, std::map<std::string, std::string>&, std::string, const TString&, const TString&);
  void FillTables(int, int, std::map<std::string, std::string>&, reco::BeamSpot&, std::string);
  void FillTrendHistos(int, int, std::map<std::string, std::string>&, reco::BeamSpot&, const TString&);
  void weight(BeamSpotMapBx&, const BeamSpotMapBx&);
  void weight(double& mean, double& meanError, const double& val, const double& valError);
  void formatFitTime(char*, const std::time_t&);

  edm::ParameterSet parameters_;
  std::string monitorName_;
  edm::InputTag bsSrc_;  // beam spot

  int fitNLumi_;
  int resetFitNLumi_;
  bool debug_;

  DQMStore* dbe_;
  BeamFitter* theBeamFitter;

  unsigned int countBx_;
  int countEvt_;   //counter
  int countLumi_;  //counter
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  int lastlumi_;   // previous LS processed
  int nextlumi_;   // next LS of Fit
  int firstlumi_;  // first LS with good fit
  int countGoodFit_;
  std::time_t refBStime[2];

  bool resetHistos_;
  bool processed_;
  // ----------member data ---------------------------
  BeamSpotMapBx fbspotMap;  //for weighted beam spots of each bunch
  std::map<std::string, std::string> varMap;
  std::map<std::string, std::string> varMap1;
  // MonitorElements:
  std::map<TString, MonitorElement*> hs;   // Tables
  std::map<TString, MonitorElement*> hst;  // Trending Histos

  //Test
  //  MonitorElement * h_x0;

  //
  std::time_t tmpTime;
  std::time_t refTime;
  std::time_t startTime;
  edm::TimeValue_t ftimestamp;
};

#endif
