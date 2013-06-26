#ifndef HFPreLightCal_H
#define HFPreLightCal_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TFile;
class TH1F;
class TH2F;

class HFPreLightCal : public edm::EDAnalyzer {
 public:
  HFPreLightCal (const edm::ParameterSet& fConfiguration);
  virtual ~HFPreLightCal ();

  // analysis itself
  virtual void analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup);

  // begin of the job
  virtual void beginJob();

  // very end of the job
  virtual void endJob(void);

 private:
  std::string histfile;
  std::string textfile;
  TFile* mFile;
  FILE* tFile;
  TH1F* hts[26][36][2];
  TH1F *htsmax,*htspinmax;
  TH1F* htspin[8][3];

  edm::InputTag hfDigiCollectionTag_;
  edm::InputTag hcalCalibDigiCollectionTag_;
};

#endif
