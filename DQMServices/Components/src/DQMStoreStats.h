#ifndef DQMStoreStats_H
#define DQMStoreStats_H

/** \class DQMStoreStats
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/18 14:52:31 $
 *  $Revision: 1.7 $
 *  \author Andreas Meyer CERN
 *  \author Jan Olzem DESY
 *   
 */

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <utility>
#include <fstream>
#include <sstream>

#include "TFile.h"
#include "TTree.h"

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declarations
//


///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsSubfolder {
 public:
  DQMStoreStatsSubfolder() { totalHistos_ = 0; totalBins_ = 0; totalMemory_ = 0; }
  std::string subfolderName_;
  unsigned int totalHistos_;
  unsigned int totalBins_;
  unsigned int totalMemory_;
  void AddBinsF( unsigned int nBins ) { ++totalHistos_; totalBins_ += nBins; totalMemory_ += ( nBins *= sizeof( float ) ); }
  void AddBinsS( unsigned int nBins ) { ++totalHistos_; totalBins_ += nBins; totalMemory_ += ( nBins *= sizeof( short ) ); }
  void AddBinsD( unsigned int nBins ) { ++totalHistos_; totalBins_ += nBins; totalMemory_ += ( nBins *= sizeof( double ) ); }
};

///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsSubsystem : public std::vector<DQMStoreStatsSubfolder> {
 public:
  DQMStoreStatsSubsystem() {}
  std::string subsystemName_;
};


///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsTopLevel : public std::vector<DQMStoreStatsSubsystem> {
 public:
  DQMStoreStatsTopLevel() {}
};




///
/// DQMStoreStats itself
///
class DQMStoreStats : public edm::EDAnalyzer {
public:
  DQMStoreStats( const edm::ParameterSet& );
  ~DQMStoreStats();

  enum statsMode { considerAllME = 0, considerOnlyLumiProductME = 1 };

protected:
   
  // BeginJob
  void beginJob();

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  // Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context);

  // DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  // Endjob
  void endJob();

private:

  int calcstats( int );
  void dumpMemoryProfile( void );
  std::pair<unsigned int, unsigned int> readMemoryEntry( void ) const;
  void print();
  
  DQMStore* dbe_;
  edm::ParameterSet parameters_;

  std::string subsystem_;
  std::string subfolder_;
  int nbinsglobal_;
  int nbinssubsys_;
  int nmeglobal_;
  int nmesubsys_;
  int maxbinsglobal_;
  int maxbinssubsys_;
  std::string maxbinsmeglobal_;
  std::string maxbinsmesubsys_;

  int statsdepth_ ;
  std::string pathnamematch_ ;
  int verbose_ ;
  
  std::vector<std::pair<time_t, unsigned int> > memoryHistoryVector_;
  time_t startingTime_;
  bool isOpenProcFileSuccessful_;
  std::stringstream procFileName_;

  bool runonendrun_ ;
  bool runonendjob_ ;
  bool runonendlumi_ ;
  bool runineventloop_ ;
  bool dumpMemHistory_;
  bool dumpToFWJR_;

  // ---------- member data ----------

};

#endif

