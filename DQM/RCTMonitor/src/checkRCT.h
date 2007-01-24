#ifndef checkRCT_h
#define checkRCT_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class TFile;
class TNtuple;

class checkRCT : public edm::EDAnalyzer {
public:
  explicit checkRCT(const edm::ParameterSet&);
  ~checkRCT();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  TFile *file;
  TNtuple *nTuple;
  std::string outputFileName;
};

#endif
