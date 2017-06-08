// -*- C++ -*-
//
// Package:     CalibForward/CTPPSPixelCalibration
// calibration
// Class  :     CTPPSPixelDAQCalibration
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Helio Nogima
//         Created:  Wed, 15 Mar 2017 02:15:07 GMT
//
#include "CondTools/CTPPS/interface/CTPPSPixelDAQCalibration.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"
#include <iostream>
#include "TFile.h"
#include "TH2F.h"

CTPPSPixelDAQCalibration::CTPPSPixelDAQCalibration(edm::ParameterSet const& conf)
{
  CalibrationFile_ = conf.getParameter<std::string>("CalibrationFile");


  fp = new TFile(CalibrationFile_.c_str());


}


CTPPSPixelDAQCalibration::~CTPPSPixelDAQCalibration()
{
  fp->Close();

  delete fp;
}

void CTPPSPixelDAQCalibration::getDAQCalibration(unsigned int detid, int row, int col, float &gain, float &pedestal){

  CTPPSPixelIndices modulepixels(156,160);
  int plane = int((detid>>16) & 0X7);
  int arm = int((detid>>24)& 0X1);
  int station = int((detid>>22)& 0X3);
  int pot = int((detid>>19)& 0X7);
  int roc = 0;
  int sector=0;
  int colROC;
  int rowROC;
  if (arm==0) sector=45;
  if (arm==1) sector=56;


  if (modulepixels.transformToROC(col,row,roc,colROC,rowROC)==0){

/// TOTEM RP numbering scheme (https://indico.cern.ch/event/626748/contributions/2531619/attachments/1438891/2214488/FRavera_CTPPSGM_April2017.pdf) 

   sprintf(pathgains,"CTPPS/CTPPS_SEC%d/CTPPS_SEC%d_RP%d%d%d/CTPPS_SEC%d_RP%d%d%d_PLN%d/CTPPS_SEC%d_RP%d%d%d_PLN%d_ROC%d_Slope2D",sector,sector,arm,station,pot,sector,arm,station,pot,plane,sector,arm,station,pot,plane,roc);
   sprintf(pathpedestals,"CTPPS/CTPPS_SEC%d/CTPPS_SEC%d_RP%d%d%d/CTPPS_SEC%d_RP%d%d%d_PLN%d/CTPPS_SEC%d_RP%d%d%d_PLN%d_ROC%d_Intercept2D",sector,sector,arm,station,pot,sector,arm,station,pot,plane,sector,arm,station,pot,plane,roc);
   gain = 0; pedestal = 0;
   TH2F* gainshisto ;
   TH2F *pedestalshisto;


   if(!(gainshisto = (TH2F*)fp->Get(pathgains))) {
  
    gain=0;
   }

   if(!(pedestalshisto = (TH2F*)fp->Get(pathpedestals))) {
  
    pedestal=0;
   }

   float slope = 0; 
   if(!gainshisto){
     slope=0;}
   else{
     slope = (gainshisto->GetBinContent(colROC+1,rowROC+1));
   }
   if (slope==0.){
     gain = 0.;
   }else{
     gain = 1./slope;
   }
   if(!pedestalshisto){
     pedestal =0;
   }else{
     pedestal = float(pedestalshisto->GetBinContent(colROC+1,rowROC+1));
   }
  delete gainshisto; delete pedestalshisto;
  }


  

  return;
}
//DEFINE_FWK_MODULE( CTPPSPixelDAQCalibration);
