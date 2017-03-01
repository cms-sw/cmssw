#ifndef ReadPixelRecHit_h
#define ReadPixelRecHit_h

/** \class ReadPixelRecHit
 *
 * ReadPixelRecHit: Example of how to read Pixel RecHits
 *
 * \author Vincenzo Chiochia 
 *
 * \version   August 2006  
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define DO_HISTO
#ifdef DO_HISTO
// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TProfile.h>
#endif

class ReadPixelRecHit : public edm::EDAnalyzer
{
 public:
  
  explicit ReadPixelRecHit(const edm::ParameterSet& conf);
  
  virtual ~ReadPixelRecHit();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();

 
 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  bool print;

#ifdef DO_HISTO
  TFile* hFile;
  TH1F *hpixid,*hpixsubid,
    *hlayerid,
    *hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;                                                                                
  TH1F *hcharge1,*hcharge2, *hcharge3;
  TH1F *hadcCharge1,*hadcCharge2, *hadcCharge3, *hadcCharge1big;
  TH1F *hxpos1,*hxpos2,*hxpos3,*hypos1,*hypos2,*hypos3;
  TH1F *hsize1,*hsize2,*hsize3,
    *hsizex1,*hsizex2,*hsizex3,
    *hsizey1,*hsizey2,*hsizey3;
 
  TH1F *hrecHitsPerDet1,*hrecHitsPerDet2,*hrecHitsPerDet3;
  TH1F *hrecHitsPerLay1,*hrecHitsPerLay2,*hrecHitsPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
 
  TH1F *hdetr, *hdetz;
  
  // Forward endcaps
  TH1F *hdetrF, *hdetzF;
  TH1F *hdisk, *hblade, *hmodule, *hpanel, *hside;
  TH1F *hcharge1F,*hcharge2F;
  TH1F *hadcCharge1F,*hadcCharge2F;
  TH1F *hxpos1F,*hxpos2F,*hypos1F,*hypos2F;
  TH1F *hsize1F,*hsize2F,
    *hsizex1F,*hsizex2F,
    *hsizey1F,*hsizey2F;
  TH1F *hrecHitsPerDet1F,*hrecHitsPerDet2F;
  TH1F *hrecHitsPerLay1F,*hrecHitsPerLay2F;
  TH1F *hdetsPerLay1F,*hdetsPerLay2F;

  TH1F *hAlignErrorX1,*hAlignErrorX2,*hAlignErrorX3;
  TH1F *hAlignErrorX4,*hAlignErrorX5,*hAlignErrorX6,*hAlignErrorX7;
  TH1F *hAlignErrorY1,*hAlignErrorY2,*hAlignErrorY3;
  TH1F *hAlignErrorY4,*hAlignErrorY5,*hAlignErrorY6,*hAlignErrorY7;
  TH1F *hErrorX1,*hErrorX2,*hErrorX3,*hErrorX4,*hErrorX5,*hErrorX6,*hErrorX7;
  TH1F *hErrorY1,*hErrorY2,*hErrorY3,*hErrorY4,*hErrorY5,*hErrorY6,*hErrorY7;
  TProfile *hErrorXB, *hErrorXF, *hErrorYB, *hErrorYF;
  TProfile *hAErrorXB, *hAErrorXF, *hAErrorYB, *hAErrorYF;

#endif


};


#endif
