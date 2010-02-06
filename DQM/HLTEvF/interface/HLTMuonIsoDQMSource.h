#ifndef HLTMuonIsoDQMSource_H
#define HLTMuonIsoDQMSource_H

/** \class HLTMuonIsoDQMSource
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

class HLTMuonIsoDQMSource : public edm::EDAnalyzer {
public:
  HLTMuonIsoDQMSource( const edm::ParameterSet& );
  ~HLTMuonIsoDQMSource();

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
  int level_;
  double coneSize_;
  edm::InputTag isolationTag_;
  edm::InputTag candCollectionTag_;

  // ----------member data ---------------------------
  bool verbose_;

  MonitorElement * hiso;
  float XMIN; float XMAX;
};

#endif

