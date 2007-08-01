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
// $Id: SiPixelGainCalibrationAnalysis.h,v 1.3 2007/06/26 14:03:33 fblekman Exp $
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


#include "CalibFormats/SiPixelObjects/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"

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


      // ----------member data ---------------------------
      // internal class for storing parameters
      class CalParameters {
      public:
	float ped;
	float gain;
      };
      class TempPixelContainer {
      public:

	uint32_t roc_id;
	uint32_t dcol_id;
	uint32_t maxrow;
	uint32_t maxcol;
	uint32_t nvcal;
	uint32_t pix_id;
	uint32_t adc;
	uint32_t row;
	uint32_t col;
	uint32_t vcal;
	uint32_t ntriggers;
	uint32_t vcal_first;
	uint32_t vcal_last;
	uint32_t vcal_step;
      };
      PixelCalib calib_; // keeps track of the calibration constants

      std::string recordName_;
      uint32_t eventno_counter_;
      std::string src_;
      std::string instance_;
      uint32_t maxNfedIds_;
      std::string inputconfigfile_;
      void fill(const TempPixelContainer & aPixel);
      void init(const TempPixelContainer & aPixel);
      // maximum numbers of columns/rows/rocs/channels
      uint32_t nrowsmax_;
      uint32_t ncolsmax_;
      uint32_t nrocsmax_;
      uint32_t nchannelsmax_;
      uint32_t vcal_fitmin_;
      uint32_t vcal_fitmax_;
      uint32_t vcal_fitmax_fixed_;
      double chisq_threshold_;
      double maximum_ped_; 
      double maximum_gain_;

      uint32_t getROCnumberfromDetIDandRowCol(uint32_t detID, uint32_t row, uint32_t col);

      // and the containers
      std::vector <PixelROCGainCalib> calib_containers_;//960 = maximum number of det IDs for pixel detector
      // tracking geometry
      edm::ESHandle<TrackerGeometry> geom_;
      
      std::map < uint32_t , uint32_t > detIDmap_;// keeps track of all used detector IDs
      int32_t detIDmap_size;

      edm::ParameterSet conf_;
      bool appendMode_;
      bool useonlyonepixel_;
      bool test_;
      SiPixelGainCalibration* SiPixelGainCalibration_; // database worker class
      SiPixelGainCalibrationService SiPixelGainCalibrationService_; // additional database worker classes
      edm::Service < TFileService >  therootfileservice_; // for saving into root files
     
      std::string fitfuncrootformula_;
};

#endif
