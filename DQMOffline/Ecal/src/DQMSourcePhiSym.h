#ifndef DQMSourcePhiSym_H
#define DQMSourcePhiSym_H

/** \class DQMSourcePhiSym
 * *
 *  DQM Source for phi symmetry stream
 *
 *  $Date: 2008/04/28  $
 *  $Revision: 1.1 $
 *  \author Stefano Argiro'
 *          Andrea Gozzelino - Università e INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

using namespace std;
using namespace edm;



// ******************************************************************
// class declaration
// ******************************************************************

class DQMSourcePhiSym : public edm::EDAnalyzer {
public:
  DQMSourcePhiSym( const edm::ParameterSet& );
  ~DQMSourcePhiSym();

protected:
   
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// method analyze
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
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events

                        
  // ************************************************
  // ----------member data ---------------------------
  // *************************************************

  
  MonitorElement * hphidistr;

  MonitorElement * hiphidistr;

  MonitorElement * hetadistr;

  MonitorElement * hietadistr;

  MonitorElement * hweightamplitude;
  MonitorElement * henergyEB;

  MonitorElement * hEventEnergy;
  MonitorElement * hEventRh;
  MonitorElement * hEventSumE;

  const CaloGeometry* geo;

};

#endif

