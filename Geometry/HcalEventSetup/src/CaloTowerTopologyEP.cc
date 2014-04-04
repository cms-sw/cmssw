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
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

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
  : m_pSet( conf ) {
 // std::cout << "CaloTowerTopologyEP::CaloTowerTopologyEP" << std::endl;
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

//  std::cout << "CaloTowerTopologyEP::produce(const HcalRecNumberingRecord& iRecord)" << std::endl;
  edm::LogInfo("HCAL") << "CaloTowerTopologyEP::produce(const HcalRecNumberingRecord& iRecord)";
  
  ReturnType myTopo(new CaloTowerTopology(&*hcaltopo));

  return myTopo;
}


