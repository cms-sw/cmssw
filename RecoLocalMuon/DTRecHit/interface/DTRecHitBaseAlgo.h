#ifndef RecoLocalMuon_DTRecHitBaseAlgo_H
#define RecoLocalMuon_DTRecHitBaseAlgo_H

/** \class DTRecHitBaseAlgo
 *  Abstract algorithmic class to compute drift distance and error 
 *  form a DT digi
 *
 *  $Date: 2007/03/10 16:14:40 $
 *  $Revision: 1.7 $
 *  \author N. Amapane & G. Cerminara - INFN Torino
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"			    

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/Common/interface/OwnVector.h"

class DTDigi;
class DTLayer;
class DTLayerId;
class DTTTrigBaseSync;

namespace edm {
  class ParameterSet;
  class EventSetup;
}





class DTRecHitBaseAlgo {

 public:
  
  /// Constructor
  DTRecHitBaseAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTRecHitBaseAlgo();
  

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup) = 0;


  /// Build all hits in the range associated to the layerId, at the 1st step.
  virtual edm::OwnVector<DTRecHit1DPair> reconstruct(const DTLayer* layer,
						     const DTLayerId& layerId,
						     const DTDigiCollection::Range& digiRange);


  /// First step in computation of Left/Right hits from a Digi.  
  /// The results are the local position (in MuBarLayer frame) of the
  /// Left and Right hit, and the error (which is common). Returns
  /// false on failure. 
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const = 0 ;


  /// Second step in hit position computation, for algorithms which support it.
  /// The impact angle is given as input, and it's used to improve the hit
  /// position (and relative error). The angle is defined in radians, with
  /// respect to the perpendicular to the layer plane. Given the local direction,
  /// angle=atan(dir.x()/-dir.z()) . This can be used when a SL segment is
  /// built, so the impact angle is known but the position along wire is not.
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       DTRecHit1D& newHit1D) const = 0;
  

  /// Third (and final) step in hits position computation, for
  /// algorithms which support it.
  /// In addition the the angle, also the global position of the hit is given
  /// as input. This allows to get the magnetic field at the hit position (and
  /// not only that at the center of the wire). Also the position along the
  /// wire is available and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire. 
  virtual bool compute(const DTLayer* layer,
		       const DTRecHit1D& recHit1D,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       DTRecHit1D& newHit1D) const = 0;

 protected:
  // The module to be used for digi time synchronization
  DTTTrigBaseSync *theSync;
  
};
#endif




