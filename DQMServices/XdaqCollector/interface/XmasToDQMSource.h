#ifndef XmasToDQMSource_H
#define XmasToDQMSource_H

/** \class XmasToDQMSource
 * *
 *  DQM Test Client
 *
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "ToDqm.h"
#include "xdata/Table.h"

//#include <set>
#include <map>

//#include <fstream>
//
// class declaration
//

struct Data 
{
    std::string lastTimestamp;
    //MonitorElement * bxHistogram1D;
    //MonitorElement * wcHistogram1D;
    MonitorElement * Histogram1D;
};

class XmasToDQMSource : public edm::EDAnalyzer {
public:
  XmasToDQMSource( const edm::ParameterSet& );
  ~XmasToDQMSource();

protected:
   
  /// BeginJob
  void beginJob();

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
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events
                        /// FIXME, make prescale module?

  // ----------member data ---------------------------

  MonitorElement * h1;
  
  
  //float XMIN; float XMAX;
  
  //std::map<std::string, MonitorElement * > HostSlotMap;
  std::map<std::string, struct Data * > HostSlotMap;
  
  std::string previousTimestamp;
  std::string NBINS;
  std::string XMIN;
  std::string XMAX;
  
  //ofstream myfile;
};

#endif

