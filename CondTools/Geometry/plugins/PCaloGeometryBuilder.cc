#include "CondTools/Geometry/plugins/PCaloGeometryBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

PCaloGeometryBuilder::PCaloGeometryBuilder( const edm::ParameterSet& iConfig )
{
}


PCaloGeometryBuilder::~PCaloGeometryBuilder()
{
}

void
PCaloGeometryBuilder::beginRun( const edm::Run&, edm::EventSetup const& es )
{
   const std::string toDB ( "_toDB" ) ;

   std::cout<<"Writing out "<<EcalBarrelGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry>       pGeb   ;
   es.get<EcalBarrelGeometry::AlignedRecord>().get(
      EcalBarrelGeometry::producerTag() + toDB, pGeb ) ;

   std::cout<<"Writing out "<<EcalEndcapGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry>       pGee   ;
   es.get<EcalEndcapGeometry::AlignedRecord>().get(
      EcalEndcapGeometry::producerTag() + toDB, pGee ) ;

   std::cout<<"Writing out "<<EcalPreshowerGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry>          pGes   ;
   es.get<EcalPreshowerGeometry::AlignedRecord>().get(
      EcalPreshowerGeometry::producerTag() + toDB, pGes ) ; 

   std::cout<<"Writing out "<<HcalGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry> pGhcal   ;
   es.get<HcalGeometry::AlignedRecord>().get(
      HcalGeometry::producerTag() + toDB, pGhcal ) ;

   std::cout<<"Writing out "<<CaloTowerGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry> pGct   ;
   es.get<CaloTowerGeometry::AlignedRecord>().get(
      CaloTowerGeometry::producerTag() + toDB, pGct ) ;

   std::cout<<"Writing out "<<ZdcGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry> pGzdc   ;
   es.get<ZdcGeometry::AlignedRecord>().get(
      ZdcGeometry::producerTag() + toDB,  pGzdc ) ;

   std::cout<<"Writing out "<<CastorGeometry::producerTag()<<std::endl ;
   edm::ESHandle<CaloSubdetectorGeometry>   pGcast   ;
   es.get<CastorGeometry::AlignedRecord>().get(
      CastorGeometry::producerTag() + toDB, pGcast ) ;
}
