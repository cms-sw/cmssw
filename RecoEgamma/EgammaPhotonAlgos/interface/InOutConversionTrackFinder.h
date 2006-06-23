#ifndef RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
#define RecoEGAMMA_ConversionTrack_InOutConversionTrackFinder_h
/** \class InOutConversionTrackFinder
 **  
 **
 **  $Id: InOutConversionTrackFinder.h,v 1.1 2006/06/09 15:50:48 nancy Exp $ 
 **  $Date: 2006/06/09 15:50:48 $ 
 **  $Revision: 1.1 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"

// C/C++ headers
#include <string>
#include <vector>

//

class MagneticField;
class CkfTrajectoryBuilder;

class InOutConversionTrackFinder : public ConversionTrackFinder {
 public :
   
   InOutConversionTrackFinder(  const edm::EventSetup& es,const edm::ParameterSet& config, const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker);
 
 virtual ~InOutConversionTrackFinder()
   {
   }



 virtual std::vector<const Trajectory*> tracks(const TrajectorySeedCollection seeds ) const;
 //virtual TrackCandidateCollection tracks(const TrajectorySeedCollection seeds ) const;
 
 private:
 edm::ParameterSet conf_;
 void initComponents();
 CkfTrajectoryBuilder*  theCkfTrajectoryBuilder_;
 std::vector<const Trajectory*> theInOutTracks_;

};

#endif
