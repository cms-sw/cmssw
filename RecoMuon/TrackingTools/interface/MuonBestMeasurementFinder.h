#ifndef RecoMuon_TrackingTools_MuonBestMeasurementFinder_H
#define RecoMuon_TrackingTools_MuonBestMeasurementFinder_H

/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *
 *  $Date: 2006/06/21 17:36:50 $
 *  $Revision: 1.3 $
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

  /// OLD ORCA algo. Reported for timing comparison pourpose
  /// Will be removed after the comparison!
  TrajectoryMeasurement* findBestMeasurement_OLD(std::vector<TrajectoryMeasurement>& measC,
						 const Propagator* propagator);
  
  const MeasurementEstimator* estimator() const { return theEstimator;}

protected:

private:

  const MeasurementEstimator* theEstimator;

};
#endif


