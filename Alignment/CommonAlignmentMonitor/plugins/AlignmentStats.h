#ifndef CommonAlignmentMonitor_AlignmentStats_H
#define CommonAlignmentMonitor_AlignmentStats_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// #include <Riostream.h>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TFile.h"
#include "TTree.h"

//using namespace edm;

class AlignmentStats: public edm::EDAnalyzer{
 public:
  AlignmentStats(const edm::ParameterSet &iConfig);
  ~AlignmentStats();
  virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
  void beginJob( );
  void endJob();

 private:


  //////inputs from config file
  edm::InputTag src_;
  edm::InputTag overlapAM_;
  bool keepTrackStats_;
  bool keepHitPopulation_;
  std::string statsTreeName_;
  std::string hitsTreeName_;
  uint32_t prescale_;
  //////
  uint32_t tmpPresc_;
  bool firstEvent_;

  //Track stats
  TFile *treefile_;
  TTree *outtree_;
  static const int MAXTRKS_=200;
  int run_, event_;
  unsigned int ntracks;
  float P[MAXTRKS_],Pt[MAXTRKS_],Eta[MAXTRKS_],Phi[MAXTRKS_],Chi2n[MAXTRKS_];
  int Nhits[MAXTRKS_][7];//0=total, 1-6=Subdets

  //Hit Population
  typedef std::map<uint32_t,uint32_t>DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;

  //  edm::ESHandle<TrackerGeometry> trackerGeometry_;
  const TrackerGeometry* trackerGeometry_;

  const edm::EventSetup *lastSetup_;

};


#endif
