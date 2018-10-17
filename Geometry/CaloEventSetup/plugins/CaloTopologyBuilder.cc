#include "Geometry/CaloEventSetup/plugins/CaloTopologyBuilder.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


CaloTopologyBuilder::CaloTopologyBuilder( const edm::ParameterSet& /*iConfig*/ )
{
   //the following line is needed to tell the framework what
   // data is being produced

// disable
//   setWhatProduced( this, &CaloTopologyBuilder::produceIdeal );
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
   edm::ESHandle<CaloGeometry>                  theGeometry   ;
   iRecord.getRecord<CaloGeometryRecord>().get( theGeometry ) ;

   ReturnType ct = std::make_unique<CaloTopology>();
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
