#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSERVICE_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSERVICE_H

//edm
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//ES Data
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h" 
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
//Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

class SiStripPedestalsService {

 public:
  SiStripPedestalsService(const edm::ParameterSet& conf);
  ~SiStripPedestalsService(){};
  
  void configure( const edm::EventSetup& es );
  void setESObjects( const edm::EventSetup& es );
  int16_t getPedestal(const uint32_t& detID,const uint32_t& strip) const;
  float getNoise (const uint32_t& detID,const uint32_t& strip) const;
  float getLowTh (const uint32_t& detID,const uint32_t& strip) const;
  float getHighTh(const uint32_t& detID,const uint32_t& strip) const;

 private:
  edm::ParameterSet conf_;
  edm::ESHandle<SiStripPedestals> ped;
  edm::ESHandle<TrackerGeometry> tkgeom;
  std::string userEnv_;  
  std::string passwdEnv_;
  bool UseCalibDataFromDB_;

  double ElectronsPerADC_;
  double ENC_;
  double BadStripProbability_;
  int    PedestalValue_;
  float  LTh_;
  float  HTh_;
};

#endif
