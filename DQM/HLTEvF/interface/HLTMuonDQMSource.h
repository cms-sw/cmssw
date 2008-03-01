#ifndef HLTMuonDQMSource_H
#define HLTMuonDQMSource_H

/** \class HLTMuonDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/02/11 17:54:14 $
 *  $Revision: 1.3 $
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

class HLTMuonDQMSource : public edm::EDAnalyzer {
public:
  HLTMuonDQMSource( const edm::ParameterSet& );
  ~HLTMuonDQMSource();

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
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events
  double coneSize_;
  edm::InputTag l2collectionTag_,l3collectionTag_,l3linksTag_,l2isolationTag_,l3isolationTag_;
  // ----------member data ---------------------------
  bool verbose_;
  bool monitorDaemon_;

  MonitorElement * hNMu[2];
  MonitorElement * hcharge[2];
  MonitorElement * hpt[2];
  MonitorElement * hptlx[2];
  MonitorElement * heta[2];
  MonitorElement * hphi[2];
  MonitorElement * hptphi[2];
  MonitorElement * hpteta[2];
  MonitorElement * hL2ptres;
  MonitorElement * hL2etares;
  MonitorElement * hL2etareseta;
  MonitorElement * hL2phires;
  MonitorElement * hL2phiresphi;
  MonitorElement * hetaphi[2];
  MonitorElement * hdr[2];
  MonitorElement * hdz[2];
  MonitorElement * hdrphi[2];
  MonitorElement * hdzeta[2];
  MonitorElement * herr0[2];
  MonitorElement * hnhit[2];
  MonitorElement * hdimumass[2];
  MonitorElement * hiso[2];

  float XMIN; float XMAX;
};

#endif

