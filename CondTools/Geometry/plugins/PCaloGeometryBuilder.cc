#include "CondTools/Geometry/plugins/PCaloGeometryBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

PCaloGeometryBuilder::PCaloGeometryBuilder( const edm::ParameterSet& iConfig )
{
}


PCaloGeometryBuilder::~PCaloGeometryBuilder()
{
}

void
PCaloGeometryBuilder::beginJob( edm::EventSetup const& es )
{
   const std::string toDB ( "_toDB" ) ;

   edm::ESHandle<CaloSubdetectorGeometry> pGhcal   ;
   es.get<HcalGeometry::AlignedRecord>().get(
      HcalGeometry::producerTag() + toDB, pGhcal ) ;

   edm::ESHandle<CaloSubdetectorGeometry>       pGeb   ;
   es.get<EcalBarrelGeometry::AlignedRecord>().get(
      EcalBarrelGeometry::producerTag() + toDB, pGeb ) ;

   edm::ESHandle<CaloSubdetectorGeometry>       pGee   ;
   es.get<EcalEndcapGeometry::AlignedRecord>().get(
      EcalEndcapGeometry::producerTag() + toDB, pGee ) ;

   edm::ESHandle<CaloSubdetectorGeometry>          pGes   ;
   es.get<EcalPreshowerGeometry::AlignedRecord>().get(
      EcalPreshowerGeometry::producerTag() + toDB, pGes ) ; 
}
