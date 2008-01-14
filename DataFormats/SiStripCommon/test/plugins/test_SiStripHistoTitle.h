// Last commit: $Id: testSiStripHistoTitle.h,v 1.2 2007/07/31 15:20:25 ratnik Exp $

#ifndef DataFormats_SiStripCommon_testSiStripHistoTitle_H
#define DataFormats_SiStripCommon_testSiStripHistoTitle_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripHistoTitle 
   @author R.Bainbridge
   @brief Simple class that tests SiStripHistoTitle.
*/
class testSiStripHistoTitle : public edm::EDAnalyzer {

 public:
  
  testSiStripHistoTitle( const edm::ParameterSet& );
  ~testSiStripHistoTitle();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripHistoTitle_H

