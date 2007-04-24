// Last commit: $Id: test_SiStripEnumsAndStrings.h,v 1.2 2007/03/21 08:23:00 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_test_SiStripEnumsAndStrings_H
#define DataFormats_SiStripCommon_test_SiStripEnumsAndStrings_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class test_SiStripEnumsAndStrings 
   @author R.Bainbridge
   @brief Simple class that tests SiStripEnumsAndStrings.
*/
class test_SiStripEnumsAndStrings : public edm::EDAnalyzer {

 public:
  
  test_SiStripEnumsAndStrings( const edm::ParameterSet& );
  ~test_SiStripEnumsAndStrings();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_test_SiStripEnumsAndStrings_H

