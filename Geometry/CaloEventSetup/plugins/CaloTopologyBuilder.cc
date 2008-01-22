#include "Geometry/CaloEventSetup/plugins/CaloTopologyBuilder.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


CaloTopologyBuilder::CaloTopologyBuilder(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloTopologyBuilder::~CaloTopologyBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTopologyBuilder::ReturnType
CaloTopologyBuilder::produce(const CaloTopologyRecord& iRecord)
{
  edm::ESHandle<CaloGeometry> theGeometry;
  std::auto_ptr<CaloTopology> pCaloTopology(new CaloTopology());
//  try 
//    {
      iRecord.getRecord<IdealGeometryRecord>().get( theGeometry );
//    }
//  catch (...) {
//    edm::LogWarning("MissingInput") << "No CaloGeometry Found found";
//  }

  //ECAL parts      
  pCaloTopology->setSubdetTopology(DetId::Ecal,EcalBarrel,new EcalBarrelTopology(theGeometry));
  pCaloTopology->setSubdetTopology(DetId::Ecal,EcalEndcap,new EcalEndcapTopology(theGeometry));
  pCaloTopology->setSubdetTopology(DetId::Ecal,EcalPreshower,new EcalPreshowerTopology(theGeometry));

  return pCaloTopology;

}
