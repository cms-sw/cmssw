#ifndef CommonDet_HICMeasurementEstimator_H
#define CommonDet_HICMeasurementEstimator_H

/** \class HICMeasurementEstimator
 *  A Chi2 Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  BoundPlane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension. Ported from ORCA.
 *
 *  $Date: 2008/05/22 15:31:36 $
 *  $Revision: 1.4 $
 *  \author todorov, cerati
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "MagneticField/Engine/interface/MagneticField.h"

class HICMeasurementEstimator : public  Chi2MeasurementEstimatorBase{
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of BoundPlane and maximalLocalDisplacement.
   */
  explicit HICMeasurementEstimator(double maxChi2, double nSigma = 3., const GeometricSearchTracker* theTracker0,
  const MagneticField * mf):
  Chi2MeasurementEstimatorBase(maxChi2,nSigma)
    {
      theTracker = theTracker0;
      bl = theTracker->barrelLayers();
      fpos = theTracker->posForwardLayers();
      fneg = theTracker->negForwardLayers();
      field = mf;
      setHICDetMap();
    }

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;
				     
  template <unsigned int D> std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;

  bool estimate( const TrajectoryStateOnSurface& ts, 
			 const BoundPlane& plane) const;
			 
  virtual std::vector<double> setCuts( Trajectory& traj, const DetLayer* theCurrentLayer ); 
  
  virtual void setHICDetMap();

  cms::HICConst* getHICConst(){return theHICConst;}

  void setHICConst(cms::HICConst* hh) {theHICConst = hh;}
  
  virtual void setLastLayer( DetLayer*&  ll){theLastLayer = ll;}; 

  virtual int getDetectorCode(const DetLayer* a);
  
  virtual void chooseCuts(int& i);
  
  virtual void setMult(int aMult=1) {theLowMult=aMult;}
  
  virtual void setSign(int& i){theSign=i;} 
  
  const MagneticField* getField() {return field;}
			 
  virtual HICMeasurementEstimator* clone() const {
    return new HICMeasurementEstimator(*this);
  }

private:
  cms::HICConst*                                     theHICConst;
  double                                             theMaxChi2;
  int                                                theNSigma;
  std::map<const DetLayer*,int>                      theBarrel;
  std::map<const DetLayer*,int>                      theForward;
  std::map<const DetLayer*,int>                      theBackward;
  
  double                                             thePhiBound;
  double                                             theZBound;
  double                                             thePhiBoundMean;
  double                                             theZBoundMean;
  const DetLayer*                                    theLastLayer;
  const DetLayer*                                    theLayer;
  const DetLayer*                                    theFirstLayer;  
  int                                                theTrajectorySize;
  double                                             thePhiWin;
  double                                             theZWin;
  double                                             thePhiWinMean;
  double                                             theZWinMean;
  
  double                                             thePhiWinB;
  double                                             theZWinB;
  double                                             thePhiWinMeanB;
  double                                             theZWinMeanB;
  
  double                                             thePhiCut;
  double                                             theZCut;
  double                                             thePhiCutMean;
  double                                             theZCutMean;
  double                                             theChi2Cut;
  double                                             theNewCut;
  double                                             theNewCutB;
// Multiplicity  
  int                                                theLowMult;
  int                                                theCutType;  
  const GeometricSearchTracker*                      theTracker; 
  std::vector<BarrelDetLayer*>                       bl;
  std::vector<ForwardDetLayer*>                      fpos;
  std::vector<ForwardDetLayer*>                      fneg;
  int                                                theSign;
  const MagneticField * field;
};

#endif
