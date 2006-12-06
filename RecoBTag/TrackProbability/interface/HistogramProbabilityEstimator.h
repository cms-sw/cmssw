#ifndef HistogramProbabilityEstimator_H
#define HistogramProbabilityEstimator_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/XMLCalibration/interface/CalibrationCategory.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogram.h"
#include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"

  /** provides the track probability to come from the primary vertex
   *  for a given track
   */

class HistogramProbabilityEstimator {

 public:


  HistogramProbabilityEstimator( AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram>  * calib3D,
                                AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram> * calib2D) 
   :   m_calibration3D(calib3D),m_calibrationTransverse(calib2D)
    {}

 ~HistogramProbabilityEstimator()
 {

  if(m_calibration3D!=0) delete m_calibration3D;
  if(m_calibrationTransverse!=0) delete m_calibrationTransverse;
 }
  pair<bool,double> probability(int ipType, float significance, const reco::Track&, const reco::Jet &, const reco::Vertex &) const;

 private:


   
 AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram> *  m_calibration3D;
 AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram> *  m_calibrationTransverse;
};

#endif







