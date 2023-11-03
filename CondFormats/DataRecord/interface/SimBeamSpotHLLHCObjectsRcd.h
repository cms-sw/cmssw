#ifndef DataRecord_SimBeamSpotHLLHCObjectsRcd_h
#define DataRecord_SimBeamSpotHLLHCObjectsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     SimBeamSpotHLLHCObjectsRcd
//
/**\class SimBeamSpotHLLHCObjectsRcd SimBeamSpotHLLHCObjectsRcd.h CondFormats/DataRecord/interface/SimBeamSpotHLLHCObjectsRcd.h

 Description: Contains the Vertex Smearing parameters used by HLLHCEvtVtxGenerator (Phase 2 BeamSpot simulation)

*/
//
// Author: Francesco Brivio (INFN Milano-Bicocca)
// Created: Thu Nov 2 2023
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class SimBeamSpotHLLHCObjectsRcd : public edm::eventsetup::EventSetupRecordImplementation<SimBeamSpotHLLHCObjectsRcd> {
};

#endif
