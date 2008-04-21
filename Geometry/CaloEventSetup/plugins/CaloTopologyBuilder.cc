#include "Geometry/CaloEventSetup/plugins/CaloTopologyBuilder.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


CaloTopologyBuilder::CaloTopologyBuilder( const edm::ParameterSet& iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this, &CaloTopologyBuilder::produceIdeal );
   setWhatProduced( this, &CaloTopologyBuilder::produceCalo  );
}


CaloTopologyBuilder::~CaloTopologyBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTopologyBuilder::ReturnType
CaloTopologyBuilder::produceCalo( const CaloTopologyRecord& iRecord )
{
   if( 0 == m_caloTopology.get() ) // ignore updates because cell list will not change!
   {
      edm::ESHandle<CaloGeometry>                  theGeometry   ;
      iRecord.getRecord<CaloGeometryRecord>().get( theGeometry ) ;

      m_caloTopology = ReturnType( new CaloTopology ) ;
      //ECAL parts      
      m_caloTopology->setSubdetTopology( DetId::Ecal,
					 EcalBarrel,
					 new EcalBarrelTopology( theGeometry ) ) ;
      m_caloTopology->setSubdetTopology( DetId::Ecal,
					 EcalEndcap,
					 new EcalEndcapTopology( theGeometry ) ) ;
      m_caloTopology->setSubdetTopology( DetId::Ecal,
					 EcalPreshower,
					 new EcalPreshowerTopology(theGeometry));   
   }
   return m_caloTopology;
}

CaloTopologyBuilder::ReturnType
CaloTopologyBuilder::produceIdeal( const IdealGeometryRecord& iRecord )
{
   edm::ESHandle<CaloGeometry> theGeometry   ;
   iRecord.get(                theGeometry ) ;

   ReturnType ct ( new CaloTopology ) ;

   //ECAL parts      
   ct->setSubdetTopology( DetId::Ecal,
			  EcalBarrel,
			  new EcalBarrelTopology( theGeometry ) ) ;

   ct->setSubdetTopology( DetId::Ecal,
			  EcalEndcap,
			  new EcalEndcapTopology( theGeometry ) ) ;

   ct->setSubdetTopology( DetId::Ecal,
			  EcalPreshower,
			  new EcalPreshowerTopology(theGeometry));   
   return ct ;
}
