#ifndef HFLightCal_H
#define HFLightCal_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TFile;
class TH1F;
class TH2F;

class HFLightCal : public edm::EDAnalyzer {
 public:
  HFLightCal (const edm::ParameterSet& fConfiguration);
  virtual ~HFLightCal ();

  // analysis itself
  virtual void analyze(const edm::Event& fEvent, const edm::EventSetup& fSetup);

  // begin of the job
  virtual void beginJob();

  // very end of the job
  virtual void endJob(void);

 private:
  std::string histfile;
  std::string textfile;
  std::string prefile;
  TFile* mFile;
  FILE* tFile;
  FILE* preFile;
  TH1F* hts[26][36][2];
  TH1F* htsm[26][36][2];
  TH1F* hsp[26][36][2];
  TH1F* hspe[26][36][2];
  TH1F* hped[26][36][2];
  TH2F *hnpemapP,*hsignalmapP,*hsignalRMSmapP,*hnpemapM,*hsignalmapM,*hsignalRMSmapM;
  TH1F *hsignalmean,*hsignalrms,*hpedmean,*hpedrms,*htmax,*htmean,*hspes,*hnpevar;
  TH1F* htspin[8][3];
  TH1F* hsppin[8][3];
  TH1F* hspepin[8][3];
  TH1F* hpedpin[8][3];
  TH1F* htsmpin[8][3];

  edm::InputTag hfDigiCollectionTag_;
  edm::InputTag hcalCalibDigiCollectionTag_;
};

#endif
