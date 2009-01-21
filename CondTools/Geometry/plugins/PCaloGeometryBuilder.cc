#include "CondTools/Geometry/plugins/PCaloGeometryBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

PCaloGeometryBuilder::PCaloGeometryBuilder( const edm::ParameterSet& iConfig )
{
}


PCaloGeometryBuilder::~PCaloGeometryBuilder()
{
}

void
PCaloGeometryBuilder::beginJob( edm::EventSetup const& es )
{
   edm::ESHandle<CaloSubdetectorGeometry> pGeb;

   es.get<EcalBarrelGeometryRecord>().get( "EcalBarrel_toDB", pGeb ) ;

   edm::ESHandle<CaloSubdetectorGeometry> pGee;

   es.get<EcalEndcapGeometryRecord>().get( "EcalEndcap_toDB", pGee ) ;

   edm::ESHandle<CaloSubdetectorGeometry> pGes;

   es.get<EcalPreshowerGeometryRecord>().get( "EcalPreshower_toDB", pGes ) ; 
}
