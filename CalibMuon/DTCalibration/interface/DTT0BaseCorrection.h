#ifndef CalibMuon_DTT0BaseCorrection_H
#define CalibMuon_DTT0BaseCorrection_H

/** \class DTT0BaseCorrection
 *  Base class to define t0 corrections
 *
 *  $Date: 2012/03/02 19:47:31 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

namespace edm {
  class EventSetup;
  class ParameterSet;
}

class DTWireId;

namespace dtCalibration {

struct DTT0Data {
public:
   // Constructor
  DTT0Data(double t0_mean, double t0_rms) : mean(t0_mean),
	                   		    rms(t0_rms) {}

  double mean;
  double rms;
}; 

class DTT0BaseCorrection {
public:
   // Constructor
   DTT0BaseCorrection();
   // Destructor
   virtual ~DTT0BaseCorrection();
   
   virtual void setES(const edm::EventSetup& setup) = 0;
   virtual DTT0Data correction(const DTWireId&) = 0;
}; 

} // namespace
#endif
