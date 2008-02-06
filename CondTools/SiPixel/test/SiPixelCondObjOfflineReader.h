#ifndef CondTools_SiPixel_SiPixelCondObjOfflineReader_H
#define CondTools_SiPixel_SiPixelCondObjOfflineReader_H
// -*- C++ -*-
//
// Package:    SiPixelCondObjOfflineReader
// Class:      SiPixelCondObjOfflineReader
// 
/**\class SiPixelCondObjOfflineReader SiPixelCondObjOfflineReader.h SiPixel/test/SiPixelCondObjOfflineReader.h

 Description: Test analyzer for reading pixel calibration from the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjOfflineReader.h,v 1.4 2006/11/09 13:20:04 chiochia Exp $
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "CondFormats/SiPixelObjOfflineects/interface/SiPixelGainCalibration.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationOfflineService.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

namespace cms{
class SiPixelCondObjOfflineReader : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjOfflineReader( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjOfflineReader(){};
  virtual void beginJob( const edm::EventSetup& );
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;

private:

  edm::ParameterSet conf_;
  edm::ESHandle<TrackerGeometry> tkgeom;
  //edm::ESHandle<SiPixelGainCalibration> SiPixelGainCalibration_;
  SiPixelGainCalibrationOfflineService SiPixelGainCalibrationService_;

  std::map< uint32_t, TH1F* >  _TH1F_Pedestals_m;
  std::map< uint32_t, TH1F* >  _TH1F_Gains_m;
  std::string filename_;
  TFile* fFile;

};
}
#endif
