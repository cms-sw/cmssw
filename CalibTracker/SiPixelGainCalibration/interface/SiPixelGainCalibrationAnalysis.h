#ifndef CalibTracker__SiPixelGainCalibrationAnalysis_H_
#define CalibTracker__SiPixelGainCalibrationAnalysis_H_
// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationAnalysis
// Class:      SiPixelGainCalibrationAnalysis
// 
/**\class SiPixelGainCalibrationAnalysis SiPixelGainCalibrationAnalysis.cc CalibTracker/SiPixelGainCalibrationAnalysis/src/SiPixelGainCalibrationAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon May  7 14:22:37 CEST 2007
// $Id: SiPixelGainCalibrationAnalysis.h,v 1.1 2007/05/20 18:08:09 fblekman Exp $
//
//



// system include files
#include <memory>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"


#include "CalibTracker/SiPixelGainCalibration/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelSLinkDataHit.h"

#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

//
// class decleration
//

class SiPixelGainCalibrationAnalysis : public edm::EDAnalyzer {
   public:
      explicit SiPixelGainCalibrationAnalysis(const edm::ParameterSet&);
      ~SiPixelGainCalibrationAnalysis();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // defined as inline below
      unsigned long long get_next_block(const unsigned char * dataptr,unsigned int i);

      // ----------member data ---------------------------
      // internal class for storing parameters
      class CalParameters {
      public:
	float ped;
	float gain;
      };
      class TempPixelContainer {
      public:
	unsigned int fed_channel;
	unsigned int roc_id;
	unsigned int dcol_id;
	int maxrow;
	int maxcol;
	unsigned int pix_id;
	unsigned int adc;
	unsigned int row;
	unsigned int col;
	unsigned int vcal;
      };
      PixelCalib *calib_; // keeps track of the calibration constants
      PixelSLinkDataHit *hitworker_; // does unpacking of raw data locally
      unsigned int datasize_;//worker that keeps track of raw data
      unsigned int datacounter_;

      unsigned int eventno_counter_;
      unsigned int maxNfedIds_;
      std::string src_;
      std::string instance_;
      std::string inputconfigfile_;
      TF1 *fitfunction;
      TF1 *fancyfitfunction;
      void fill(const TempPixelContainer & aPixel);
      void init(const TempPixelContainer & aPixel);
      // maximum numbers of columns/rows/rocs/channels
      unsigned int nrowsmax_;
      unsigned int ncolsmax_;
      unsigned int nrocsmax_;
      unsigned int nchannelsmax_;
      unsigned int vcal_fitmin_;
      unsigned int vcal_fitmax_;
      unsigned int vcal_fitmax_fixed_;
      double chisq_threshold_;
      double maximum_gain_;
      double maximum_ped_;

      unsigned int getROCnumberfromDetIDandRowCol(unsigned int detID, unsigned int row, unsigned int col);

      // and the containers
      //  bool rocgainused_[40*24];// [channel][roc]
      PixelROCGainCalib calib_containers_[40*24];// [channel][roc]
      // tracking geometry
      edm::ESHandle<TrackerGeometry> geom_;
      
      std::map < unsigned int , unsigned int > detIDmap_;// keeps track of all used detector IDs
      
      unsigned int detIDmap_size;
      unsigned int getindexfromdetid(unsigned int detid);
      edm::ParameterSet conf_;
      bool appendMode_;
      SiPixelGainCalibration* SiPixelGainCalibration_; // database worker class
      SiPixelGainCalibrationService SiPixelGainCalibrationService_; // additional database worker classes
      std::string recordName_;
      //
/*       double thefancyfitfunction(double *x, double *par);// 0: pedestal; 1:plateau; 2:halfwaypoint; 3:gain */
};

inline unsigned int SiPixelGainCalibrationAnalysis::getindexfromdetid(unsigned int detid){
  if(detIDmap_[detid]==0){// entry does not exist
    detIDmap_[detid]=detIDmap_size;
    detIDmap_size++;
   
  }
  return detIDmap_[detid];
}
inline unsigned long long SiPixelGainCalibrationAnalysis::get_next_block(const unsigned char * dataptr,unsigned int i){
  return ((const unsigned long long*)dataptr)[i];
}

// fancy fit function
/* inline Double_t SiPixelGainCalibrationAnalysis::thefancyfitfunction(Double_t *x, Double_t *par){// 0: pedestal; 1:plateau; 2:halfwaypoint; 3:gain */
/*   Double_t ped = par[0]; */
/*   Double_t plateau = par[1]; */
/*   Double_t halfwaypoint = par[2]; */
/*   Double_t gain = par[3]; */
  
/*   Double_t res = ped; */
/*   Double_t turnon = 1+TMath::Erf((x[0]-halfwaypoint)/(gain*sqrt(x[0]))); */
/*   turnon*=0.5*plateau; */
  
/*   return res+turnon; */
/* } */
#endif
