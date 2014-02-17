#ifndef ElectronSeedAnalyzer_h
#define ElectronSeedAnalyzer_h

//
// Package:         RecoEgamma/ElectronTrackSeed
// Class:           ElectronSeedAnalyzer
//

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronSeedAnalyzer.h,v 1.6 2012/09/13 20:08:32 wdd Exp $
//
//


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField;
class TFile;
class TH1F;
class TH1I;
class TTree;

class ElectronSeedAnalyzer : public edm::EDAnalyzer
{
 public:

  explicit ElectronSeedAnalyzer( const edm::ParameterSet & conf );
  virtual ~ElectronSeedAnalyzer();
  virtual void analyze( const edm::Event &, const edm::EventSetup &);
  virtual void beginJob();
  virtual void endJob();

 private:

  TrajectoryStateTransform transformer_;

  TFile *histfile_;
  TTree *tree_;
  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];
  int seedSubdet1[10], seedSubdet2[10];
  int seedLayer1[10], seedLayer2[10];
  int seedSide1[10], seedSide2[10];
  float seedDphi1[10], seedDrz1[10], seedDphi2[10], seedDrz2[10];
  float seedPhi1[10], seedRz1[10], seedPhi2[10], seedRz2[10];
  TH1F *histeMC_;
  TH1F *histeMCmatched_;
  TH1F *histecaldriveneMCmatched_;
  TH1F *histtrackerdriveneMCmatched_;
  TH1F *histp_;
  TH1F *histeclu_;
  TH1F *histpt_;
  TH1F *histptMC_;
  TH1F *histptMCmatched_;
  TH1F *histecaldrivenptMCmatched_;
  TH1F *histtrackerdrivenptMCmatched_;
  TH1F *histetclu_;
  TH1F *histeffpt_;
  TH1F *histeta_;
  TH1F *histetaMC_;
  TH1F *histetaMCmatched_;
  TH1F *histecaldrivenetaMCmatched_;
  TH1F *histtrackerdrivenetaMCmatched_;
  TH1F *histetaclu_;
  TH1F *histeffeta_;
  TH1F *histq_;
  TH1F *histeoverp_;
  TH1I *histnrseeds_;
  TH1I *histnbseeds_;
  TH1I *histnbclus_;

  edm::InputTag inputCollection_;
  edm::InputTag beamSpot_;
//  std::vector<std::pair<const GeomDet*, TrajectoryStateOnSurface> >  mapTsos_;
//  std::vector<std::pair<std::pair<const GeomDet*,GlobalPoint>,  TrajectoryStateOnSurface> >  mapTsos2_;

 };

#endif



