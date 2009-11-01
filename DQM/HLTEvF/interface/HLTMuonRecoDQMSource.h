#ifndef HLTMuonRecoDQMSource_H
#define HLTMuonRecoDQMSource_H

/** \class HLTMuonRecoDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/06/25 10:46:58 $
 *  $Revision: 1.1 $
 *  \author  M. Vander Donckt CERN
 *   
 */
#include <memory>
#include <unistd.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class HLTMuonRecoDQMSource : public edm::EDAnalyzer {
public:
  HLTMuonRecoDQMSource( const edm::ParameterSet& );
  ~HLTMuonRecoDQMSource();

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
 
  edm::ParameterSet parameters_;

  DQMStore* dbe_;  
  std::string monitorName_;
  std::string outputFile_;
  int level_;
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events
  double coneSize_;
  edm::InputTag candCollectionTag_;
  edm::InputTag beamSpotTag_;
  edm::InputTag l2seedscollectionTag_;
  // ----------member data ---------------------------
  bool verbose_;

  MonitorElement * hNMu;
  MonitorElement * hcharge;
  MonitorElement * hpt;
  MonitorElement * heta;
  MonitorElement * hphi;
  MonitorElement * hptphi;
  MonitorElement * hpteta;
  MonitorElement * hptres;
  MonitorElement * hetares;
  MonitorElement * hetareseta;
  MonitorElement * hphires;
  MonitorElement * hphiresphi;
  MonitorElement * hetaphi;
  MonitorElement * hdr;
  MonitorElement * hd0;
  MonitorElement * hdz;
  MonitorElement * hdrphi;
  MonitorElement * hd0phi;
  MonitorElement * hdzeta;
  MonitorElement * herr0;
  MonitorElement * hnhit;
  MonitorElement * hdimumass;
  float XMIN; float XMAX;
};

#endif

