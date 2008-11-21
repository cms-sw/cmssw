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
// $Id: L1RCTGenCalibrator.h,v 1.6 2008/08/19 20:22:46 lgray Exp $
//
//

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCalibrator.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraphAsymmErrors.h"
#include "TGraph2DErrors.h"
#include "TF1.h"
#include "TF2.h"
#include "TVectorT.h"

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
    
  class event_data
  {
  public: 
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

namespace root_structs
{
  struct Event
  {
    unsigned event, run;
  };

  struct Generator
  {
    unsigned nGen;
    int particle_type[100];
    double et[100],eta[100],phi[100];
    unsigned crate[100],card[100],region[100];
  };

  struct Region
  {
    unsigned nRegions;
    int linear_et[200], ieta[200],iphi[200];
    double eta[200],phi[200];
    unsigned crate[200],card[200],region[200];
  };
  
  struct TPG
  {
    unsigned nTPG;
    int ieta[3100],iphi[3100];
    double eta[3100],phi[3100],ecalEt[3100],hcalEt[3100],ecalE[3100],hcalE[3100];
    unsigned crate[3100],card[3100],region[3100];
  };

}


#endif
