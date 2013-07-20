#ifndef RecoMuon_L3TrackFinder_MuonRoadTrajectoryBuilder_h
#define RecoMuon_L3TrackFinder_MuonRoadTrajectoryBuilder_h

/** \file RecoMuon/L3TrackFinder/interface/GlobalMuonRoadTrajectoryBuilder.h
 *  This class provides trajectory building from a TrajectorySeed defined on an
 *  inner layer of the tracker detector, no RecHit is required on the TrajectorySeed.
 *  Combinatorics between RecHits is made. RecHits are accessed via the MeasurementTracker
 *  
 *  $Date: 2012/12/26 21:32:53 $
 *  $Revision: 1.6 $
 *  \author Adam Evertt, Jean-Roch Vlimant
 */



#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
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

#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"


#include <vector>

class MuonRoadTrajectoryBuilder :public TrajectoryBuilder {
 public:
  ///constructor from PSet and things from record
   MuonRoadTrajectoryBuilder(const edm::ParameterSet & par,const MeasurementTracker * mt,const MagneticField * f,const Propagator * p);

   //// destructor
   ~MuonRoadTrajectoryBuilder();
     
   ////setup the event. required by MeasurementTracker to get strip/pixel clusters
   void setEvent(const edm::Event&) const ;
     
   ////process the seed
   std::vector<Trajectory> trajectories(const TrajectorySeed & seed) const ;

   /// process the seed, in a faster manner.
   void trajectories(const TrajectorySeed & seed, TrajectoryContainer &ret) const ; 
	 
 private:
	 
   ///Info/Debug category "Muon|RecoMuon|MuonRoadTrajectoryBuilder"
   std::string theCategory ;
   
 public:
   ///for the trajectory collection
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
   
   template <class A> class flippingPair : public std::pair<A,A>{
   public:
     flippingPair(): theFlip(false) {;}
     void flip(){theFlip=!theFlip;}
     A & head(){ if (theFlip) return this->first ;else return this->second;}
     A & tail(){if (theFlip) return this->second ;else return this->first;}
   private:
     bool theFlip;
   };
   
   typedef std::list< trajectory > TrajectoryCollection;
   typedef flippingPair<TrajectoryCollection> TrajectoryCollectionFPair ;
   
   //  const MuonServiceProxy * theProxyService;
   
   ////internal method to produce trajectories
   void makeTrajectories(const TrajectorySeed & seed, std::vector<Trajectory> & result ,int version=0) const ;
   ////internal method that use how grown trajectory builder
   void makeTrajectories_0(const TrajectorySeed & seed, std::vector<Trajectory> & result) const ;
   ////internal method to grow trajectories on one tracker layer
   int GatherHits(const TrajectoryStateOnSurface & step,const DetLayer * thislayer, TrajectoryCollectionFPair & Trajectories) const ;
   ////internal method not yet implemented
   void makeTrajectories_1(const TrajectorySeed & seed, std::vector<Trajectory> & result) const ;
	  
   ////check and reduce the trajectory according to number of trajectories found
   bool checkStep(TrajectoryCollection & collection) const ;
   ////check an remove trajectories subset of each other. keeping the longest
   void checkDuplicate(TrajectoryCollection & collection) const ;
	      
   Trajectory smooth(Trajectory &) const ;
   void cleanTrajectory(Trajectory & traj) const ;
	      
   //only global scope variable
   mutable bool theFirstlayer;
   //algorithm options
   //limit the total number of possible trajectories taken into account for a single seed
   unsigned int theMaxTrajectories;
	      
   //limit the type of module considered to gather rechits
   ////theDynamicMaxNumberOfHitPerModule=false: theNumberOfHitPerModule is used always
   ////theDynamicMaxNumberOfHitPerModule=true: theNumberOfHitPerModule is used first, then theNumberOfHitPerModuleThreshold[0] after
   ////the number of trajectory is more than theMaxTrajectoriesThreshold[0]. and so on ...
   bool theDynamicMaxNumberOfHitPerModule;
   unsigned int theNumberOfHitPerModuleDefault;
   mutable unsigned int theNumberOfHitPerModule;
   std::vector<unsigned int> theMaxTrajectoriesThreshold;
   std::vector<unsigned int> theNumberOfHitPerModuleThreshold;
  
   //fixed parameters
   bool theBranchonfirstlayer;
   bool theCarriedIPatfirstlayer;
   bool theCarriedIPatfirstlayerModule;
  
   //output track candidate selection
   ////minimum number of hit require per track candidate
   unsigned int  theMinNumberOfHitOnCandidate;
   ////select to output all possible trajectories found in the event
   bool theOutputAllTraj;
  
  
   //tools and usefull pointers
   ////measurement tracke handle
   const MeasurementTracker * theMeasurementTracker;
    
   //kalman tools
   ////chi2 governed in by cfg file, Nsig = sqrt(chi2)
   Chi2MeasurementEstimator * theRoadEstimator;
   ////chi2 governed in by cfg file, Nsig = sqrt(chi2)
   Chi2MeasurementEstimator * theHitEstimator;
   ////instantiate with a KFupdator
   TrajectoryStateUpdator * theUpdator;
   ////instantiate with a KFTrajectorySmoother
   KFTrajectorySmoother * theSmoother;
   
   ////trajectory transformer
   
   
   ////magnetic field handle
   const MagneticField * theField;
   
   ////propagator handle
   const Propagator * thePropagator;
   
};


#endif
