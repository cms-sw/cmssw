/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/MeasurementDet/interface/MeasurementMuonSystem.h"

MeasurementMuonSystem::MeasurementMuonSystem(){}


MeasurementMuonSystem::~MeasurementMuonSystem(){}


const MeasurementDet* 
MeasurementMuonSystem::idToDet(const DetId& id) const {
  return 0;
}

