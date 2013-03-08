// Last commit: $Id $

#ifndef CalibFormats_SiStripObjects_test_SiStripQuality_H
#define CalibFormats_SiStripObjects_test_SiStripQuality_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class test_SiStripQuality 
   @author L.Quertenmont
   @brief Simple class that tests SiStripQuality.
*/
class testSiStripQuality : public edm::EDAnalyzer {

 public:
  testSiStripQuality( const edm::ParameterSet& ){;}
  ~testSiStripQuality(){;}
  void analyze( const edm::Event&, const edm::EventSetup& );
};

#endif // CalibFormats_SiStripObjects_test_SiStripQuality_H

