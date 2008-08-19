#ifndef __RCT_GENCALIBRATOR_H__
#define __RCT_GENCALIBRATOR_H__
// -*- C++ -*-
//
// Package:    L1RCTCalibrator
// Class:      L1RCTCalibrator
//
/**\class L1RCTCalibrator L1RCTCalibrator.cc src/L1RCTCalibrator/src/L1RCTCalibrator.cc

 Description: Analyzer that calibrates to Generator level data.

 Implementation:

*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTGenCalibrator.h,v 1.5 2008/08/14 14:20:27 lgray Exp $
//
//

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCalibrator.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraphAsymmErrors.h"
#include "TGraph2DErrors.h"
#include "TF1.h"
#include "TF2.h"

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "RooGlobalFunc.h"

// forward declarations
namespace reco
{
  class GenParticle;
}

using namespace RooFit;

//
// class declaration
//
class L1RCTGenCalibrator : public L1RCTCalibrator
{
 public:

  typedef TH1F* TH1Fptr;
  typedef TH2F* TH2Fptr;
  typedef TGraphAsymmErrors* TGraphptr;
  typedef TGraph2DErrors* TGraph2Dptr;
  typedef RooDataSet* pRooDataSet;
  typedef RooRealVar* pRooRealVar;
  
  // condensed data structs
  class generator
  {
  public:
    int particle_type;
    double et, phi, eta;
    rct_location loc;

    bool operator==(const generator& r) const { return ( particle_type == r.particle_type && et == r.et &&
							 phi == r.phi && eta == r.eta ); }
  };
    
  struct event_data
  {
    unsigned event;
    unsigned run;
    std::vector<generator> gen_particles;
    std::vector<region> regions;
    std::vector<tpg> tpgs;
  };

  explicit L1RCTGenCalibrator(edm::ParameterSet const&);
  ~L1RCTGenCalibrator();

  void saveCalibrationInfo(const view_vector&,const edm::Handle<ecal_view>&, 
			   const edm::Handle<hcal_view>&, const edm::Handle<reg_view>&);
  void postProcessing();
  
  void bookHistograms(); 

private:
  // ----------private member functions---------------
  void saveGenInfo(const reco::GenParticle*, const edm::Handle<ecal_view>&, const edm::Handle<hcal_view>&,
		   const edm::Handle<reg_view>&, std::vector<generator>*, std::vector<region>*,
		   std::vector<tpg>*);

  std::vector<generator> overlaps(const std::vector<generator>&) const;

  // vector of all event data
  std::vector<event_data> data_;

 public:
  // histograms

  // diagnostic histograms
  TH1Fptr hEvent, hRun, hGenPhi, hGenEta, hGenEt, hGenEtSel, 
    hRCTRegionEt, hRCTRegionPhi, hRCTRegionEta,
    hTpgSumEt, hTpgSumEta, hTpgSumPhi;
  
  TH2Fptr hGenPhivsTpgSumPhi, hGenEtavsTpgSumEta, hGenPhivsRegionPhi, hGenEtavsRegionEta;

  TH1Fptr hDeltaEtPeakvsEtaBin_uc[12], hDeltaEtPeakvsEtaBin_c[12], hDeltaEtPeakRatiovsEtaBin[12], 
    hDeltaEtPeakvsEtaBinAllEt_uc, hDeltaEtPeakvsEtaBinAllEt_c, hDeltaEtPeakRatiovsEtaBinAllEt;

  TH1Fptr hPhotonDeltaR95[28], hNIPionDeltaR95[28], hPionDeltaR95[28] ;
  
  // histograms for algorithm
  TGraphptr gPhotonEtvsGenEt[28], gNIPionEtvsGenEt[28];
  TGraph2Dptr gPionEcalEtvsHcalEtvsGenEt[28];

  TH1Fptr hPhotonDeltaEOverE[28], hPionDeltaEOverE[28];

  //RooDataSets, since TGraphs suck
  pRooRealVar roorvPhotonGenEt[28], roorvPhotonTPGSumEt[28], 
    roorvNIPionGenEt[28], roorvNIPionTPGSumEt[28];
  pRooDataSet roodsPhotonEtvsGenEt[28], roodsNIPionEtvsGenEt[28];
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

#endif
