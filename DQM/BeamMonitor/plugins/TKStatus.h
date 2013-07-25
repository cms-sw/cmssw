#ifndef TKStatus_H
#define TKStatus_H

/** \class TKStatus
 * *
 *  $Date: 2010/06/03 21:03:51 $
 *  $Revision: 1.2 $
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *   
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>


//
// class declaration
//

class TKStatus : public edm::EDAnalyzer {
 public:
  TKStatus( const edm::ParameterSet& );
  ~TKStatus();

 protected:
   
  // BeginJob
  void beginJob();

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;
  
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			    const edm::EventSetup& context) ;
  
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
			  const edm::EventSetup& c);  
  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  // Endjob
  void endJob(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  
 private:

  void dumpTkDcsStatus(std::string &);

  edm::ParameterSet parameters_;
  std::string dcsTkFileName_;
  std::ofstream fasciiDcsTkFile;

  bool debug_;
  bool checkStatus_;
  int countEvt_;       //counter
  int countLumi_;      //counter
  int beginLumi_;
  int endLumi_;
  int lastlumi_; // previous LS processed
  bool dcsTk[6];
  // ----------member data ---------------------------
  
  //
  std::time_t tmpTime;
  std::time_t refTime;
  edm::TimeValue_t ftimestamp;
  int runnum;
};

#endif

