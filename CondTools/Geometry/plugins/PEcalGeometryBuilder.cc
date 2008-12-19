#include "PEcalGeometryBuilder.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/PEcalGeometry.h"
#include "Geometry/Records/interface/PEcalBarrelRcd.h"
#include "Geometry/Records/interface/PEcalEndcapRcd.h"
#include "Geometry/Records/interface/PEcalPreshowerRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"

#include <iostream>

PEcalGeometryBuilder::PEcalGeometryBuilder( const edm::ParameterSet& iConfig )
{
}


PEcalGeometryBuilder::~PEcalGeometryBuilder()
{
}

void
PEcalGeometryBuilder::beginJob(edm::EventSetup const& es )
{
  using namespace edm;
  
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("PEcalGeometryBuilder")<<"PoolDBOutputService unavailable";
    return;
  }
  
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG);     

  const CaloSubdetectorGeometry* geomEB (pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)); 
  const CaloSubdetectorGeometry* geomEE (pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)); 
  const CaloSubdetectorGeometry* geomES (pG->getSubdetectorGeometry(DetId::Ecal,EcalPreshower)); 
    
  geomEB->getSummary(m_transEB,m_indEB,m_dimEB);  
  geomEE->getSummary(m_transEE,m_indEE,m_dimEE);  
  geomES->getSummary(m_transES,m_indES,m_dimES);  

  PEcalGeometry* pebg = new PEcalGeometry(m_transEB,m_dimEB,m_indEB);  
  PEcalGeometry* peeg = new PEcalGeometry(m_transEE,m_dimEE,m_indEE);  
  PEcalGeometry* pesg = new PEcalGeometry(m_transES,m_dimES,m_indES);  

  if ( mydbservice->isNewTagRequest("PEcalBarrelRcd") ) {
    mydbservice->createNewIOV<PEcalGeometry>(pebg,mydbservice->beginOfTime(),mydbservice->endOfTime(),"PEcalBarrelRcd");
  } else {
    mydbservice->appendSinceTime<PEcalGeometry>(pebg,mydbservice->currentTime(),"PEcalBarrelRcd");
  }
  if ( mydbservice->isNewTagRequest("PEcalEndcapRcd") ) {
    mydbservice->createNewIOV<PEcalGeometry>(peeg,mydbservice->beginOfTime(),mydbservice->endOfTime(),"PEcalEndcapRcd");
  } else {
    mydbservice->appendSinceTime<PEcalGeometry>(peeg,mydbservice->currentTime(),"PEcalEndcapRcd");
  }
  if ( mydbservice->isNewTagRequest("PEcalPreshowerRcd") ) {
    mydbservice->createNewIOV<PEcalGeometry>(pesg,mydbservice->beginOfTime(),mydbservice->endOfTime(),"PEcalPreshowerRcd");
  } else {
    mydbservice->appendSinceTime<PEcalGeometry>(pesg,mydbservice->currentTime(),"PEcalPreshowerRcd");
  }

}
