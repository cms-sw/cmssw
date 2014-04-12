
#include "testEcalTPGScale.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"


testEcalTPGScale::testEcalTPGScale(edm::ParameterSet const& pSet)
{
  std::cout<<"I'm going to check the internal consistancy of EcalTPGScale transformation..."<<std::endl ;
}

void testEcalTPGScale::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) 
{
  using namespace edm;
  using namespace std;

  // geometry
  ESHandle<CaloGeometry> theGeometry;
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<CaloGeometryRecord>().get( theGeometry );
  evtSetup.get<EcalEndcapGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  evtSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get(eTTmap_);
  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);

  EcalTPGScale ecalScale ;
  ecalScale.setEventSetup(evtSetup) ;

  bool error(false) ;
  vector<DetId>::const_iterator it ;

  // EB
  const std::vector<DetId>& ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  it = ebCells.begin() ;
  const EBDetId idEB(*it) ;
  const EcalTrigTowerDetId towidEB = idEB.tower();
  for (unsigned int ADC=0 ; ADC<256 ; ADC++) {
    double gev = ecalScale.getTPGInGeV(ADC, towidEB) ;
    unsigned int tpgADC = ecalScale.getTPGInADC(gev, towidEB) ;
    if (tpgADC != ADC) {
      error = true ;
      cout<<" ERROR : with ADC = "<<ADC<<" getTPGInGeV = "<<gev<<" getTPGInADC = "<<tpgADC<<endl ;
    }
    ecalScale.getLinearizedTPG(ADC, towidEB) ;
  }

  // EE
  const std::vector<DetId>& eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  it = eeCells.begin() ;
  const EEDetId idEE(*it);
  const EcalTrigTowerDetId towidEE = (*eTTmap_).towerOf(idEE) ;
  for (unsigned int ADC=0 ; ADC<256 ; ADC++) {
    double gev = ecalScale.getTPGInGeV(ADC, towidEE) ;
    unsigned int tpgADC = ecalScale.getTPGInADC(gev, towidEE) ;
    if (tpgADC != ADC) {
      error = true ;
      cout<<" ERROR : with ADC = "<<ADC<<" getTPGInGeV = "<<gev<<" getTPGInADC = "<<tpgADC<<endl ;
    }
    ecalScale.getLinearizedTPG(ADC, towidEE) ;
  }


  if (!error) cout<<" there is no error with EcalTPGScale internal consistancy "<<endl ;

}

void testEcalTPGScale::beginJob()
{
  using namespace edm;
  using namespace std;

}

