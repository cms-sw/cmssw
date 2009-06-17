#ifndef DQM_HLTEVF_HLTMONMUONCLIENT_H
#define DQM_HLTEVF_HLTMONMUONCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class HLTMonMuonClient: public edm::EDAnalyzer {
public:

  /// Constructor
  HLTMonMuonClient(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~HLTMonMuonClient();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  edm::ParameterSet parameters;

  static const int NTRIG = 20;

  int nTriggers_;

  DQMStore* dbe_;
  std::string indir_, outdir_;

  std::vector<std::string> theHLTCollectionLabels;
  std::vector<std::string> theHLTCollectionLevel;
  std::vector<edm::InputTag> theHLTCollectionL1seed;
  std::vector<edm::InputTag> theHLTCollectionL1filter;
  std::vector<edm::InputTag> theHLTCollectionL2filter;
  std::vector<edm::InputTag> theHLTCollectionL2isofilter;
  std::vector<edm::InputTag> theHLTCollectionL3filter;
  std::vector<edm::InputTag> theHLTCollectionL3isofilter;

  MonitorElement * hEffSummary;
  MonitorElement * hCountSummary;
  MonitorElement * hSubFilterEfficiency[NTRIG];
  MonitorElement * hSubFilterCount[NTRIG];
};

#endif

