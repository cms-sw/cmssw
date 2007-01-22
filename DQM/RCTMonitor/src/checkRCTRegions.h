#ifndef checkRCTRegions_h
#define checkRCTRegions_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TFile;
class TNtuple;

class checkRCTRegions : public edm::EDAnalyzer {
public:
  explicit checkRCTRegions(const edm::ParameterSet&);
  ~checkRCTRegions();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  TFile *file;
  TNtuple *nTuple;
};

#endif
