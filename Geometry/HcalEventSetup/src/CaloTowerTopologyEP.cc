// -*- C++ -*-
//
// Package:    CaloTowerTopologyEP
// Class:      CaloTowerTopologyEP
// 
/**\class CaloTowerTopologyEP CaloTowerTopologyEP.h tmp/CaloTowerTopologyEP/interface/CaloTowerTopologyEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

#include "Geometry/HcalEventSetup/interface/CaloTowerTopologyEP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloTowerTopologyEP::CaloTowerTopologyEP(const edm::ParameterSet& conf)
{
  edm::LogInfo("HCAL") << "CaloTowerTopologyEP::CaloTowerTopologyEP";
  setWhatProduced(this);
}


CaloTowerTopologyEP::~CaloTowerTopologyEP() { 
}

void CaloTowerTopologyEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription desc;
  descriptions.add( "CaloTowerTopology", desc );
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTowerTopologyEP::ReturnType
CaloTowerTopologyEP::produce(const HcalRecNumberingRecord& iRecord) {
  edm::ESHandle<HcalTopology> hcaltopo;
  iRecord.get(hcaltopo);

  edm::LogInfo("HCAL") << "CaloTowerTopologyEP::produce(const HcalRecNumberingRecord& iRecord)";

  return std::make_unique<CaloTowerTopology>(&*hcaltopo);
}


