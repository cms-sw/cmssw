#ifndef RecoMuon_TrackingTools_MuonBestMeasurementFinder_H
#define RecoMuon_TrackingTools_MuonBestMeasurementFinder_H

/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *  the chi2 cut for the MeasurementEstimator is huge since should not be used.
 *  The aim of this class is to return the "best" measurement according to the
 *  chi2, but without any cut. The decision whether to use or not the
 *  measurement is taken in the caller class.
 *  The evaluation is made (in hard-code way) with the granularity = 1. Where
 *  the granularity is the one defined in the MuonTrajectoyUpdatorClass.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

#include <vector>

class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class MuonTransientTrackingRecHit;

class MuonBestMeasurementFinder {
  typedef std::vector<TrajectoryMeasurement*>    TMContainer;
  typedef TMContainer::iterator                 TMIterator;

public:
  
  /// Constructor
  MuonBestMeasurementFinder();
  
  /// Destructor
  virtual ~MuonBestMeasurementFinder();

  // Operations

   /// return the Tm with the best chi2: no cut applied.
  TrajectoryMeasurement* findBestMeasurement(std::vector<TrajectoryMeasurement>& measC,
					     const Propagator* propagator);
  
  std::pair<double,int> lookAtSubRecHits(TrajectoryMeasurement* measurement,
					 const Propagator* propagator);

  const MeasurementEstimator* estimator() const { return theEstimator;}

protected:

private:

  const MeasurementEstimator* theEstimator;

};
#endif


