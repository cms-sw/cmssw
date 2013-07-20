#ifndef CalibMuon_DTTTrigBaseCorrection_H
#define CalibMuon_DTTTrigBaseCorrection_H

/** \class DTTTrigBaseCorrection
 *  Base class to define the tTrig corrections for entering in DB
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.3 $
 *  \author A. Vilela Pereira
 */

namespace edm {
  class EventSetup;
  class ParameterSet;
}

class DTSuperLayerId;

namespace dtCalibration {

struct DTTTrigData {
public:
   // Constructor
  DTTTrigData(double ttrig_mean, double ttrig_sigma, double kFact) : mean(ttrig_mean),
								     sigma(ttrig_sigma),
								     kFactor(kFact) {}

  double mean;
  double sigma;
  double kFactor;
}; 

class DTTTrigBaseCorrection {
public:
   // Constructor
   DTTTrigBaseCorrection();
   // Destructor
   virtual ~DTTTrigBaseCorrection();
   
   virtual void setES(const edm::EventSetup& setup) = 0;
   virtual DTTTrigData correction(const DTSuperLayerId&) = 0;
};
 
} // namespace
#endif
