#ifndef GlobalMuonRSTrajectoryBuilder_h
#define GlobalMuonRSTrajectoryBuilder_h


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/SiStripDetId/interface/TIBDetId.h>
#include <DataFormats/SiStripDetId/interface/TOBDetId.h>
#include <DataFormats/SiStripDetId/interface/TIDDetId.h>
#include <DataFormats/SiStripDetId/interface/TECDetId.h>
#include <DataFormats/TrackReco/interface/Track.h>

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <DataFormats/GeometryVector/interface/GlobalVector.h>
#include <DataFormats/GeometrySurface/interface/BoundCylinder.h>
#include <DataFormats/GeometrySurface/interface/BoundDisk.h>

#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h>
#include <TrackingTools/PatternTools/interface/Trajectory.h>
#include <TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h>
#include <TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h>
#include <TrackingTools/PatternTools/interface/TrajectorySmoother.h>
#include <TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include <TrackingTools/DetLayers/interface/DetLayer.h>

#include <MagneticField/Engine/interface/MagneticField.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"


#include <vector>

class MuonServiceProxy;

#ifndef O
#define O(var) #var<<": "<<var<<"\n"
#endif
#ifndef On
#define On(var) #var<<": "<<var<<" "
#endif


#ifndef Df
#define Df(var,os) os<<#var<<" "<<var<<"\n";
#endif
#ifndef Dfn
#define Dfn(var,os) os<<#var<<" "<<var<<" ";
#endif

/*
  #ifndef Dfl
  #define Dfl(label,var,o) o<<label<<" "<<var<<"\n";
  #endif
  #ifndef Dfln
  #define Dfln(label,var,o) o<<label<<" "<<var;
  #endif*/


#ifndef D
#define D(var) std::cout<<#var<<" "<<var<<"\n";
#endif
#ifndef Dn
#define Dn(var) std::cout<<#var<<" "<<var<<" ";
#endif
#ifndef Dl
#define Dl(label,var) std::cout<<label<<" "<<var<<std::endl;
#endif
#ifndef Dln
#define Dln(label,var) std::cout<<label<<" "<<var;
#endif
#ifndef COMMENT
#define COMMENT(text) std::cout<<"### "<<text<<std::endl;
#endif
#ifndef COMMENTf
#define COMMENTf(text,o) o<<"### "<<text<<"\n";
#endif


template <class A> class flippingPair : public std::pair<A,A>{
 public:
  flippingPair(): _flip(false) {;}
  void flip(){_flip=!_flip;}
  A & head(){ if (_flip) return this->first ;else return this->second;}
  A & tail(){if (_flip) return this->second ;else return this->first;}
 private:
  bool _flip;
};

class GlobalMuonRSTrajectoryBuilder {
 public:
  GlobalMuonRSTrajectoryBuilder(const edm::ParameterSet & par);
  ~GlobalMuonRSTrajectoryBuilder();

  void init(const MuonServiceProxy*);  
  void setEvent(const edm::Event&);
  void set(const edm::EventSetup&);

  //process the seeds for one STA muon
  std::vector<Trajectory> trajectories(const TrajectorySeed & seeds);

  bool setDebug(bool b){bool sb=_debug;_debug=b;return sb;}
 private:
  
  std::string _category;
  //  inline std::string & category(){ return _category;}
  //to be removed maybe
  void Show(const DetId det,std::ostream & o=std::cout);
  void Show(const TrajectoryStateOnSurface & TSOS, char * label,std::ostream & o=std::cout);
  void Show(const FreeTrajectoryState & FS,char * label,std::ostream & o=std::cout);
  std::string Modulename(const DetId det);
  
 public:
  //for the hit collection only
  class trajectory {
  public:
    trajectory():duplicate(false),chi2(0),missed(0),lastmissed(false),missedinarow(0){;}
    bool duplicate;
    double chi2;
    int missed;
    bool lastmissed;
    int missedinarow;
    std::list <TrajectoryMeasurement > measurements;
    TrajectoryStateOnSurface TSOS;
    Trajectory traj; //the only thing to keep eventually
  };
 private:
  typedef std::list< trajectory > TrajectoryCollection;
  //  bool _trajectorysource;
  //  TrajectoryCollection _theTrajectories[2];
  typedef flippingPair<TrajectoryCollection> TrajectoryCollectionFPair ;
  //  TrajectoryCollectionFPair _thePTrajectories; //no global scope container

  const MuonServiceProxy * theProxyService;

  //internale method to produce trajectories
  void makeTrajectories(const TrajectorySeed & seed, std::vector<Trajectory> & result ,int version=0);
  void makeTrajectories_0(const TrajectorySeed & seed, std::vector<Trajectory> & result);
  int GatherHits(const TrajectoryStateOnSurface & step,const DetLayer * thislayer, TrajectoryCollectionFPair & Trajectories);
  void makeTrajectories_1(const TrajectorySeed & seed, std::vector<Trajectory> & result);

  //check the content of trajectories
  bool checkStep(TrajectoryCollection & collection);
  void checkDuplicate(TrajectoryCollection & collection);

  Trajectory smooth(Trajectory &);
  void cleanTrajectory(Trajectory & traj);

  //only global scope variable
  bool _firstlayer;
  //algorithm options
  //limit the total number of possible trajectories taken into account for a single seed
  uint _maxTrajectories;

  //limit the type of module considered to gather rechits
  //_dynamicMaxNumberOfHitPerModule=false: _numberOfHitPerModule is used always
  //_dynamicMaxNumberOfHitPerModule=true: _numberOfHitPerModule is used first, then _numberOfHitPerModuleThreshold[0] after
  //the number of trajectory is more than _maxTrajectoriesThreshold[0]. and so on ...
  bool _dynamicMaxNumberOfHitPerModule;
  uint _numberOfHitPerModule;
  std::vector<uint> _maxTrajectoriesThreshold;
  std::vector<uint> _numberOfHitPerModuleThreshold;
  
  //fixed parameters
  bool _branchonfirstlayer;
  bool _carriedIPatfirstlayer;
  bool _carriedIPatfirstlayerModule;
  
  //output track candidate selection
  unsigned int  _minNumberOfHitOnCandidate;
  bool _outputAllTraj;
  
  //debug flag
  bool _debug;

  
  //tools and usefull pointers
  //geometry tools
  edm::ESHandle<MeasurementTracker> _measurementTracker;
    
  //kalman tools
  Chi2MeasurementEstimator * _roadEstimator;//chi2 governed in by cfg file, Nsig = sqrt(chi2)
  Chi2MeasurementEstimator * _hitEstimator;//chi2 governed in by cfg file, Nsig = sqrt(chi2)
  TrajectoryStateUpdator * _updator;//instantiate with a KFupdator
  //  TrajectorySmoother * _smoother;//instantiate with a KFTrajectorySmoother
  KFTrajectorySmoother * _smoother;//instantiate with a KFTrajectorySmoother
  
  //trajectory transformer
  TrajectoryStateTransform _transformer;
  
  //magnetic field
  edm::ESHandle<MagneticField> _field;
  
  //propagator
  std::string _propagatorName;
  edm::ESHandle<Propagator> _prop;

};


#endif
