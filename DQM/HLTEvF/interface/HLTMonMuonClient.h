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

  void initialize();
  edm::ParameterSet parameters;

  DQMStore* dbe;
  std::string input_dir, output_dir;
  //int counterLS;      ///counter
  //int counterEvt;     ///counter
  //int prescaleLS;     ///units of lumi sections
  //int prescaleEvt;    ///prescale on number of events

  // -------- member data --------
  //MonitorElement *csctferrors_;
};

#endif

