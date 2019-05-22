#include "Geometry/CaloEventSetup/plugins/CaloTopologyBuilder.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


CaloTopologyBuilder::CaloTopologyBuilder( const edm::ParameterSet& /*iConfig*/ ):
  geometryToken_{setWhatProduced(this, &CaloTopologyBuilder::produceCalo).consumesFrom<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})}
{}

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
   const auto& geometry = iRecord.get(geometryToken_);

   ReturnType ct = std::make_unique<CaloTopology>();
   //ECAL parts      
   ct->setSubdetTopology( DetId::Ecal,
			  EcalBarrel,
			  std::make_unique<EcalBarrelTopology>( geometry ) ) ;
   ct->setSubdetTopology( DetId::Ecal,
			  EcalEndcap,
			  std::make_unique<EcalEndcapTopology>( geometry ) ) ;
   ct->setSubdetTopology( DetId::Ecal,
			  EcalPreshower,
			  std::make_unique<EcalPreshowerTopology>());
   return ct ;
}
