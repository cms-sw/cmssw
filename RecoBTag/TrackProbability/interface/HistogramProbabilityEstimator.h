#ifndef HistogramProbabilityEstimator_H
#define HistogramProbabilityEstimator_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/BTagObjects/interface/CalibratedHistogram.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/TrackProbability/interface/CalibrationInterface.h"
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

   HistogramProbabilityEstimator( AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>  * calib3Dold,
                                 AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML> * calib2Dold)
  {
/*    m_calibration3D = new    CalibrationInterface<TrackClassFilterCategory,CalibratedHistogram>;
    for(size_t i=0;i<ca3D->data.size(); i++)
     {
        cout <<  "  Adding category" << endl;
        calib3d->addEntry(TrackClassFilterCategory(ca3D->data[i].category),ca3D->data[i].histogram); // convert category data to filtering category
     }
 
   m_calibrationTransverse = new    CalibrationInterface<TrackClassFilterCategory,CalibratedHistogram>;
*/
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







