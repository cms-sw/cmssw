// Last commit: $Id: test_Template.h,v 1.1 2007/04/24 12:20:00 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_test_Template_H
#define DQM_SiStripCommissioningClients_test_Template_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class test_Template 
   @author R.Bainbridge
   @brief Simple class that tests Template.
*/
class test_Template : public edm::EDAnalyzer {

 public:
  
  test_Template( const edm::ParameterSet& );
  ~test_Template();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}

 private:
  
};

#endif // DQM_SiStripCommissioningClients_test_Template_H

