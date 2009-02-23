#ifndef DQMStoreStats_H
#define DQMStoreStats_H

/** \class DQMStoreStats
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/02/23 12:59:24 $
 *  $Revision: 1.1 $
 *  \author Andreas Meyer CERN
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class DQMStoreStats : public edm::EDAnalyzer {
public:
  DQMStoreStats( const edm::ParameterSet& );
  ~DQMStoreStats();

protected:
   
  // BeginJob
  void beginJob(const edm::EventSetup& c);

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

  int calcstats();
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

  // ---------- member data ----------

};

#endif

