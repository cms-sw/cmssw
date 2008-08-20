#ifndef HcalTPGAnalyzer_h
#define HcalTPGAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTPGAnalyzer : public edm::EDAnalyzer {
public:
  explicit HcalTPGAnalyzer(const edm::ParameterSet&);
  ~HcalTPGAnalyzer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();
private:
};


#endif
