#ifndef MILLEPEDEMONITOR_H
#define MILLEPEDEMONITOR_H

/// \class MillePedeMonitor
///
/// monitoring of MillePedeAlignmentAlgorithm and its input tracks
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision$
///  $Date$
///  (last update by $Author$)

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h" // Algebraic matrices

#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

/***************************************
****************************************/
class MillePedeMonitor
{
 public:
  // book histograms in constructor
  MillePedeMonitor(const char *rootFile = "trackMonitor.root");
  MillePedeMonitor(TDirectory *rootDir);
  // writes histograms in destructor
  ~MillePedeMonitor();

  void fillTrack(const reco::Track *track, const Trajectory *traj);
  void fillRefTrajectory(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr);
  void fillDetLabel(unsigned int globalDetLabel);
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

  TH1         *myDetLabelBitHist;
  TH1         *myDetLabelHist;

};

template <class OBJECT_TYPE>  
int MillePedeMonitor::GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name)
{
  int result = 0;
  for (typename std::vector<OBJECT_TYPE*>::const_iterator iter = vec.begin();
       iter != vec.end(); ++iter, ++result) {
    if (*iter && (*iter)->GetName() == name) return result;
  }
  edm::LogError("Alignment") << "@SUBMillePedeMonitor::GetIndex" << " could not find " << name;
  return -1;
}
#endif
