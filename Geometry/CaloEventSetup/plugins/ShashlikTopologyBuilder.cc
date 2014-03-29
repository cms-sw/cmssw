#include "Geometry/CaloEventSetup/plugins/ShashlikTopologyBuilder.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


ShashlikTopologyBuilder::ShashlikTopologyBuilder(const edm::ParameterSet& /*iConfig*/) {
  //the following line is needed to tell the framework what
  // data is being produced

  setWhatProduced( this);
}


ShashlikTopologyBuilder::~ShashlikTopologyBuilder() { }


//
// member functions
//

// ------------ method called to produce the data  ------------
ShashlikTopologyBuilder::ReturnType
ShashlikTopologyBuilder::produce(const ShashlikNumberingRecord& iRecord ) {

  edm::ESHandle<ShashlikDDDConstants>  pSDC;
  iRecord.get( pSDC ) ;
  const ShashlikDDDConstants* sdc = &(*pSDC);

  ReturnType ct ( new ShashlikTopology(sdc) ) ;
  std::cout << "Create ShashlikTopology(sdc)" << std::endl;
  return ct ;
}
