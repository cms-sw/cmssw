#ifndef HLTMuonDQMSource_H
#define HLTMuonDQMSource_H

/** \class HLTMuonDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/05 09:54:04 $
 *  $Revision: 1.5 $
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
  edm::InputTag l2collectionTag_;
  edm::InputTag l2isolationTag_;
  edm::InputTag l2seedscollectionTag_;
  edm::InputTag l3collectionTag_;
  edm::InputTag l3isolationTag_;
  //  edm::InputTag l3linksTag_;

  // ----------member data ---------------------------
  bool verbose_;

  MonitorElement * hNMu[4];
  MonitorElement * hcharge[4];
  MonitorElement * hpt[4];
  MonitorElement * hptlx[2];
  MonitorElement * heta[4];
  MonitorElement * hphi[4];
  MonitorElement * hptphi[4];
  MonitorElement * hpteta[4];
  MonitorElement * hptres[3];
  MonitorElement * hetares[3];
  MonitorElement * hetareseta[3];
  MonitorElement * hphires[3];
  MonitorElement * hphiresphi[3];
  MonitorElement * hetaphi[4];
  MonitorElement * hdr[2];
  MonitorElement * hd0[2];
  MonitorElement * hdz[2];
  MonitorElement * hdrphi[2];
  MonitorElement * hd0phi[2];
  MonitorElement * hdzeta[2];
  MonitorElement * herr0[2];
  MonitorElement * hnhit[4];
  MonitorElement * hdimumass[2];
  MonitorElement * hiso[2];
  MonitorElement * hl1quality;
  float XMIN; float XMAX;
};

#endif

