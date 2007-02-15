#ifndef HistogramProbabilityEstimator_H
#define HistogramProbabilityEstimator_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
//#include "RecoBTag/TrackProbability/interface/CalibrationInterface.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationInterface.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"


 #include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"

  /** provides the track probability to come from the primary vertex
   *  for a given track
   */
#include <utility>

class HistogramProbabilityEstimator {

 public:


  HistogramProbabilityEstimator( CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML>  * calib3D,
                                CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML> * calib2D) 
   :   m_calibration3D(calib3D),m_calibrationTransverse(calib2D)
    {}

   HistogramProbabilityEstimator( AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>  * calib3D,
                                 AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML> * calib2D)
    :   m_calibration3D(calib3D),m_calibrationTransverse(calib2D)

  {
  }

 ~HistogramProbabilityEstimator()
 {

  if(m_calibration3D!=0) delete m_calibration3D;
  if(m_calibrationTransverse!=0) delete m_calibrationTransverse;
 }
  std::pair<bool,double> probability(int ipType, float significance, const reco::Track&, const reco::Jet &, const reco::Vertex &) const;

 private:
   CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML> *  m_calibration3D;
   CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML> *  m_calibrationTransverse;
};

#endif







