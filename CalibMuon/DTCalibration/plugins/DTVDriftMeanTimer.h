#ifndef CalibMuon_DTCalibration_DTVDriftMeanTimer_h
#define CalibMuon_DTCalibration_DTVDriftMeanTimer_h

/** \class DTVDriftMeanTimer
 *  Concrete implementation of a DTVDriftBaseAlgo.
 *  Computes vDrift using the Mean Timer algorithm.
 *
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TFile;
class DTMeanTimerFitter;

namespace dtCalibration {

class DTVDriftMeanTimer: public DTVDriftBaseAlgo {
public:
   DTVDriftMeanTimer(edm::ParameterSet const&);
   virtual ~DTVDriftMeanTimer();

   virtual void setES(const edm::EventSetup& setup);
   virtual DTVDriftData compute(const DTSuperLayerId&);
private:
   TFile* rootFile_;
   DTMeanTimerFitter* fitter_;
};

} // namespace
#endif
