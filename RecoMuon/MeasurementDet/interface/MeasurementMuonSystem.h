#ifndef MeasurementDet_MeasurementMuonSystem_h
#define MeasurementDet_MeasurementMuonSystem_h

/** \class MeasurementMuonSystem
 *
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"

class MeasurementMuonSystem : public MeasurementDetSystem {
 public:
  /// Constructor
  MeasurementMuonSystem();

  /// Destructor
  virtual ~MeasurementMuonSystem();
  
  // Return the pointer to the MeasurementDet corresponding to a given DetId
  virtual const MeasurementDet* idToDet(const DetId& id) const;

 private:

};
#endif

