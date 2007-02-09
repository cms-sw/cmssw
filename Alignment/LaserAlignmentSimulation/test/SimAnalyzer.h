/* 
 * Get some statistics and plots about the simulation of the Laser Alignment System
 */

#ifndef LaserAlignmentSimulation_SimAnalyzer_H
#define LaserAlignmentSimulation_SimAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Surface/interface/BoundSurface.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

// ROOT
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include "TFile.h"
#include "TObject.h"

#include <iostream>

class SimAnalyzer : public edm::EDAnalyzer, public TObject 
{
 public:
  explicit SimAnalyzer(edm::ParameterSet const& theConf);
  ~SimAnalyzer();
  
  // this method will do the user analysis 
  virtual void analyze(edm::Event const& theEvent, edm::EventSetup const& theSetup);
  virtual void beginJob(const edm::EventSetup& theSetup);
    
 private:
  double angle(double theAngle);

  void closeRootFile();
  
  void initHistograms();

  void trackerStatistics(edm::Event const& theEvent, edm::EventSetup const& theSetup);
  
 private:
  int theEvents;
  int theDebugLevel;
  double theSearchPhiTIB;
  double theSearchPhiTOB;
  double theSearchPhiTEC;
  double theSearchZTIB;
  double theSearchZTOB;

  // Tree stuff
  TFile * theFile;
  int theCompression;
  std::string theFileName;

  // the histograms for Barrel Hits
  TH1D * theBarrelSimHitsX;
  TH1D * theBarrelSimHitsY;
  TH1D * theBarrelSimHitsZ;
  TH2D * theBarrelSimHitsYvsX;
  TH2D * theBarrelSimHitsXvsZ;
  TH2D * theBarrelSimHitsYvsZ;
  TH2D * theBarrelSimHitsRvsZ;
  TH2D * theBarrelSimHitsPhivsX;
  TH2D * theBarrelSimHitsPhivsY;
  TH2D * theBarrelSimHitsPhivsZ;

  // the histograms for Endcap Hits
  TH1D * theEndcapSimHitsX;
  TH1D * theEndcapSimHitsY;
  TH1D * theEndcapSimHitsZ;
  TH2D * theEndcapSimHitsYvsX;
  TH2D * theEndcapSimHitsXvsZ;
  TH2D * theEndcapSimHitsYvsZ;
  TH2D * theEndcapSimHitsRvsZ;
  TH2D * theEndcapSimHitsPhivsX;
  TH2D * theEndcapSimHitsPhivsY;
  TH2D * theEndcapSimHitsPhivsZ;

  // the histograms for all SimHits
  TH2D * theSimHitsRvsZ;
  TH2D * theSimHitsPhivsZ;  
};
#endif
