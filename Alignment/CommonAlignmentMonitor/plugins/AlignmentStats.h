#ifndef CommonAlignmentMonitor_AlignmentStats_H
#define CommonAlignmentMonitor_AlignmentStats_H


#include <Riostream.h>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"


#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Utilities/General/interface/ClassName.h"

#include "TFile.h"
#include "TTree.h"

//using namespace edm;

class AlignmentStats: public edm::EDAnalyzer{
 public:
  AlignmentStats(const edm::ParameterSet &iConfig);
  ~AlignmentStats();
  virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
  void beginJob( const edm::EventSetup &iSetup  );
  void endJob();

 private:


  //////inputs from config file
  edm::InputTag src_;
  edm::InputTag overlapAM_;
  bool keepTrackStats_;
  bool keepHitPopulation_;
  std::string statstreename_;
  std::string hitstreename_;
  uint32_t prescale_;
  //////
  uint32_t tmppresc;


  //Track stats
  TFile *treefile_;
  TTree *outtree_;
  static const int MAXTRKS=200;
  int run, event;
  unsigned int ntracks;
  float P[MAXTRKS],Pt[MAXTRKS],Eta[MAXTRKS],Phi[MAXTRKS],Chi2n[MAXTRKS];
  int Nhits[MAXTRKS][7];//0=total, 1-6=Subdets

  //Hit Population
  typedef map<uint32_t,uint32_t>DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;

  edm::ESHandle<TrackerGeometry> trackerGeometry_;
};


#endif
