#ifndef MILLEPEDEMONITOR_H
#define MILLEPEDEMONITOR_H

/// \class MillePedeMonitor
///
/// monitoring of MillePedeAlignmentAlgorithm and its input tracks
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.1 $
///  $Date: 2006/10/20 13:44:13 $
///  (last update by $Author: flucke $)

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h" // Algebraic matrices

#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>
#include <TString.h>
#include <TH1.h>
#include <TH2.h>

class TH1;
class TH2;
class TDirectory;
class Trajectory;

namespace reco {
  class Track;
}

class Alignable;
class AlignableDet;


/***************************************
****************************************/
class MillePedeMonitor
{
 public:
  // book histograms in constructor
  MillePedeMonitor(const char *rootFile = "trackMonitor.root");
  MillePedeMonitor(TDirectory *rootDir);
  // writes histograms in destructor
  ~MillePedeMonitor(); // non-virtual destructor: not intended to be parent class

  void fillTrack(const reco::Track *track, const Trajectory *traj);
  void fillRefTrajectory(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr);
  void fillMille(const TransientTrackingRecHit::ConstRecHitPointer &recHit,
		 const std::vector<float> &localDerivs, 
		 const std::vector<float> &globalDerivs,
		 float residuum, float sigma);
  void fillFrameToFrame(AlignableDet *aliDet, Alignable *ali);

 private:
  bool init(TDirectory *directory);
  bool equidistLogBins(double* bins, int nBins, double first, double last) const;

  template <class OBJECT_TYPE>  
    int GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name);

  TDirectory *myRootDir;
  bool        myDeleteDir; 

  std::vector<TH1*> myTrackHists1D;
  std::vector<TH1*> myTrajectoryHists1D;
  std::vector<TH2*> myTrajectoryHists2D;
  std::vector<TH2*> myMilleHists2D;
  std::vector<TH2*> myFrame2FrameHists2D;

};

template <class OBJECT_TYPE>  
int MillePedeMonitor::GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name)
{
  int result = 0;
  for (typename std::vector<OBJECT_TYPE*>::const_iterator iter = vec.begin();
       iter != vec.end(); ++iter, ++result) {
    if (*iter && (*iter)->GetName() == name) return result;
  }
  edm::LogError("Alignment") << "@SUB=MillePedeMonitor::GetIndex" << " could not find " << name;
  return -1;
}
#endif
