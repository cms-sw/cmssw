#ifndef ERRORMATRIC_H
#define ERRORMATRIC_H


#include <TRandom2.h>
#include <TFile.h>
#include <TProfile3D.h>
#include <TString.h>

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//purpose
//provice a parametrisation of curvilinear error matrix
//can smear a free trajectory state according to the error matrix

class ErrorMatrix{
 public:
  enum action { use , constructor};
  ErrorMatrix(const edm::ParameterSet & pset);
  
  ~ErrorMatrix();
  void close();
  
  CurvilinearTrajectoryError get(GlobalVector momentum);
  //  CurvilinearTrajectoryError get_random(GlobalVector momentum);
  inline TProfile3D * get(int i , int j) {return theIndex(i,j);}
  inline uint index(int i, int j){return thePindex(i,j);}

  //  TrajectoryStateOnSurface Randomize(const TrajectoryStateOnSurface & tsos,double scaleFactor=1)
  //  FreeTrajectoryState Randomize(const FreeTrajectoryState & fts,double scaleFactor=1)

  static const TString vars[5];

 private:
  TFile * theF;
  TProfile3D * theData[15];
  //x axis- pT
  //y axis- eta
  //z axis- phi
  
  inline int thePindex(int i , int j) {
    static const int offset[5]={0,5,5+4,5+4+3,5+4+3+2};
    return offset[i]+abs(j-i);}

  inline TProfile3D * theIndex(int i , int j) {
    return theData[thePindex(i,j)];}

  double theValue(GlobalVector & momentum, int i, int j);
  double theRms(GlobalVector & momentum, int i, int j);
  
  std::string theCategory;
};



#endif
