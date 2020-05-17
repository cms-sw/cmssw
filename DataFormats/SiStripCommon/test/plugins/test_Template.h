
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
  test_Template(const edm::ParameterSet&);
  ~test_Template() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }

private:
};

#endif  // DQM_SiStripCommissioningClients_test_Template_H
