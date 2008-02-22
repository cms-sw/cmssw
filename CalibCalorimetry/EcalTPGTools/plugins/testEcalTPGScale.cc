
#include "testEcalTPGScale.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

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

  EcalTPGScale ecalScale ;
  bool error(false) ;
  vector<DetId>::const_iterator it ;

  // EB
  std::vector<DetId> ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  it = ebCells.begin() ;
  const EBDetId idEB(*it) ;
  const EcalTrigTowerDetId towidEB = idEB.tower();
  int RCT_LUT_EB[256] ;
  for (uint ADC=0 ; ADC<256 ; ADC++) {
    double gev = ecalScale.getTPGInGeV(evtSetup, ADC, towidEB) ;
    uint tpgADC = ecalScale.getTPGInADC(evtSetup, gev, towidEB) ;
    if (tpgADC != ADC) {
      error = true ;
      cout<<" ERROR : with ADC = "<<ADC<<" getTPGInGeV = "<<gev<<" getTPGInADC = "<<tpgADC<<endl ;
    }
    RCT_LUT_EB[ADC] = ecalScale.getLinearizedTPG(evtSetup, ADC, towidEB) ;
  }

  // EE
  std::vector<DetId> eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  it = eeCells.begin() ;
  const EEDetId idEE(*it);
  const EcalTrigTowerDetId towidEE = (*eTTmap_).towerOf(idEE) ;
  int RCT_LUT_EE[256] ;
  for (uint ADC=0 ; ADC<256 ; ADC++) {
    double gev = ecalScale.getTPGInGeV(evtSetup, ADC, towidEE) ;
    uint tpgADC = ecalScale.getTPGInADC(evtSetup, gev, towidEE) ;
    if (tpgADC != ADC) {
      error = true ;
      cout<<" ERROR : with ADC = "<<ADC<<" getTPGInGeV = "<<gev<<" getTPGInADC = "<<tpgADC<<endl ;
    }
    RCT_LUT_EE[ADC] = ecalScale.getLinearizedTPG(evtSetup, ADC, towidEE) ;
  }


  if (!error) cout<<" there is no error with EcalTPGScale internal consistancy "<<endl ;

}

void testEcalTPGScale::beginJob(const edm::EventSetup& evtSetup)
{
  using namespace edm;
  using namespace std;

  // geometry
  ESHandle<CaloGeometry> theGeometry;
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<IdealGeometryRecord>().get( theGeometry );
  evtSetup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get(eTTmap_);
  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);

}

