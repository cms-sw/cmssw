#ifndef CalibTracker_SiStripCommon_testSiStripFedIdListReader_h
#define CalibTracker_SiStripCommon_testSiStripFedIdListReader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <string>

/**
   @class testSiStripFedIdListReader
   @author R.Bainbridge
*/
class testSiStripFedIdListReader : public edm::EDAnalyzer {
  
 public:
  
  explicit testSiStripFedIdListReader( const edm::ParameterSet& );
  ~testSiStripFedIdListReader() {;}
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:

  edm::FileInPath fileInPath_;
  
};

#endif

