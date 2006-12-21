#ifndef checkTPGs_h
#define checkTPGs_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class checkTPGs : public edm::EDAnalyzer {
public:
  explicit checkTPGs(const edm::ParameterSet&);
  ~checkTPGs();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
};

#endif
