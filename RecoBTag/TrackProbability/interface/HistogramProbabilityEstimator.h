#ifndef HistogramProbabilityEstimator_H
#define HistogramProbabilityEstimator_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

// #include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"

  /** provides the track probability to come from the primary vertex
   *  for a given track
   */
#include <utility>

class HistogramProbabilityEstimator {

 public:


  HistogramProbabilityEstimator( const  TrackProbabilityCalibration  * calib3D,
                                const TrackProbabilityCalibration * calib2D) 
   :   m_calibration3D(calib3D),m_calibration2D(calib2D)
    {}

/*   HistogramProbabilityEstimator( AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>  * calib3D,
                                 AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML> * calib2D)
    :   m_calibration3D(calib3D),m_calibrationTransverse(calib2D)

  {
  }
*/

 ~HistogramProbabilityEstimator()
 {

//  if(m_calibration3D!=0) delete m_calibration3D;
//  if(m_calibration2D!=0) delete m_calibration2D;
 }
  std::pair<bool,double> probability(bool quality, int ipType, float significance, const reco::Track&, const reco::Jet &, const reco::Vertex &) const;

 private:
  const TrackProbabilityCalibration * m_calibration3D;
 const TrackProbabilityCalibration * m_calibration2D;
   
};

#endif







