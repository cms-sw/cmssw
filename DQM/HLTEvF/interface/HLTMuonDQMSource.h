#ifndef HLTMuonDQMSource_H
#define HLTMuonDQMSource_H

/** \class HLTMuonDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/01/24 14:43:19 $
 *  $Revision: 1.1 $
 *  \author  M. Zanetti CERN
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
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

  DaqMonitorBEInterface* dbe_;  
  std::string monitorName_;
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events
                        /// FIXME, make prescale module?
  double coneSize_;
  edm::InputTag l2collectionTag_,l3collectionTag_,l3linksTag_,l2isolationTag_,l3isolationTag_;
  // ----------member data ---------------------------

  MonitorElement * hL2NMu;
  MonitorElement * hL2charge;
  MonitorElement * hL2pt;
  MonitorElement * hL2ptlx;
  MonitorElement * hL2eta;
  MonitorElement * hL2phi;
  MonitorElement * hL2ptres;
  MonitorElement * hL2etares;
  MonitorElement * hL2phires;
  MonitorElement * hL2etaphi;
  MonitorElement * hL2dr;
  MonitorElement * hL2dz;
  MonitorElement * hL2err0;
  MonitorElement * hL2nhit;
  MonitorElement * hL2iso;
  MonitorElement * hL2dimumass;
  MonitorElement * hL3NMu;
  MonitorElement * hL3charge;
  MonitorElement * hL3pt;
  MonitorElement * hL3ptlx;
  MonitorElement * hL3eta;
  MonitorElement * hL3phi;
  MonitorElement * hL3etaphi;
  MonitorElement * hL3dr;
  MonitorElement * hL3dz;
  MonitorElement * hL3err0;
  MonitorElement * hL3nhit;
  MonitorElement * hL3iso;
  MonitorElement * hL3dimumass;

  float XMIN; float XMAX;
};

#endif

