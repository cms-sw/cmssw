/** 
 * Demo analyzer for reading digis
 * author A.Tumanov 2/22/06 
 *   
 */

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


class DigiAnalyzer : public edm::EDAnalyzer {
public:
  explicit DigiAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  //
  int eventNumber;
};



