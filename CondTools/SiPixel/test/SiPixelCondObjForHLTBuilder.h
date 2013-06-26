#ifndef CondTools_SiPixel_SiPixelCondObjForHLTBuilder_H
#define CondTools_SiPixel_SiPixelCondObjForHLTBuilder_H
// -*- C++ -*-
//
// Package:    SiPixelCondObjForHLTBuilder
// Class:      SiPixelCondObjForHLTBuilder
// 
/**\class SiPixelCondObjForHLTBuilder SiPixelCondObjForHLTBuilder.h SiPixel/test/SiPixelCondObjForHLTBuilder.h

 Description: Test analyzer for writing pixel calibration in the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjForHLTBuilder.h,v 1.7 2010/01/12 11:29:54 rougny Exp $
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include <string>

namespace cms{
class SiPixelCondObjForHLTBuilder : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjForHLTBuilder( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjForHLTBuilder(){};
  virtual void beginJob();
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;
  bool loadFromFile();

private:

  edm::ParameterSet conf_;
  bool appendMode_;
  SiPixelGainCalibrationForHLT* SiPixelGainCalibration_;
  SiPixelGainCalibrationForHLTService SiPixelGainCalibrationService_;
  std::string recordName_;

  double meanPed_;
  double rmsPed_;
  double meanGain_;
  double rmsGain_;
  double meanPedFPix_;
  double rmsPedFPix_;
  double meanGainFPix_;
  double rmsGainFPix_;
  double deadFraction_;
  double noisyFraction_;
  double secondRocRowGainOffset_;
  double secondRocRowPedOffset_;
  int numberOfModules_;
  bool fromFile_;
  std::string fileName_; 
  bool generateColumns_;

  // Internal class
  class CalParameters {
  public:
    float p0;
    float p1;
  };
  // Map for storing calibration constants
  std::map<int,CalParameters, std::less<int> > calmap_;
  PixelIndices * pIndexConverter_; // Pointer to the index converter 

};
}
#endif
