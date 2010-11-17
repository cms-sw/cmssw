#ifndef CalibMuon_DTCalibration_DTVDriftBaseAlgo_h
#define CalibMuon_DTCalibration_DTVDriftBaseAlgo_h

/** \class DTVDriftBaseAlgo
 *  Base class to define algorithm for vDrift computation 
 *
 *  $Date: 2008/12/11 16:34:34 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

namespace edm {
  class EventSetup;
  class ParameterSet;
}

class DTSuperLayerId;

struct DTVDriftData {
public:
  DTVDriftData(double vdrift_mean, double vdrift_sigma): mean(vdrift_mean), sigma(vdrift_sigma) {}

  double mean;
  double sigma;
}; 

class DTVDriftBaseAlgo {
public:
   DTVDriftBaseAlgo();
   virtual ~DTVDriftBaseAlgo();
   
   virtual void setES(const edm::EventSetup& setup) = 0;
   virtual DTVDriftData compute(const DTSuperLayerId&) = 0;
}; 

#endif
