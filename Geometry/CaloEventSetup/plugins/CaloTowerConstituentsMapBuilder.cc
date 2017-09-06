// -*- C++ -*-
//
// Package:    CaloTowerConstituentsMapBuilder
// Class:      CaloTowerConstituentsMapBuilder
// 
/**\class CaloTowerConstituentsMapBuilder CaloTowerConstituentsMapBuilder.h tmp/CaloTowerConstituentsMapBuilder/interface/CaloTowerConstituentsMapBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/CaloEventSetup/plugins/CaloTowerConstituentsMapBuilder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <zlib.h>
#include <strings.h>

//
// constructors and destructor
//
CaloTowerConstituentsMapBuilder::CaloTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig) :
  mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile","")),
  mapAuto_(iConfig.getUntrackedParameter<bool>("MapAuto",false)),
  skipHE_(iConfig.getUntrackedParameter<bool>("SkipHE",false)) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

//now do what ever other initialization is needed
}


CaloTowerConstituentsMapBuilder::~CaloTowerConstituentsMapBuilder() {}


//
// member functions
//

void
CaloTowerConstituentsMapBuilder::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "MapFile", "" );
  desc.addUntracked<bool>( "MapAuto", false );
  desc.addUntracked<bool>( "SkipHE", false );
  descriptions.add( "caloTowerConstituents", desc );
}

// ------------ method called to produce the data  ------------
CaloTowerConstituentsMapBuilder::ReturnType
CaloTowerConstituentsMapBuilder::produce(const CaloGeometryRecord& iRecord)
{
  edm::ESHandle<HcalTopology> hcaltopo;
  iRecord.getRecord<HcalRecNumberingRecord>().get(hcaltopo);

  edm::ESHandle<CaloTowerTopology> cttopo;
  iRecord.getRecord<HcalRecNumberingRecord>().get(cttopo);

  auto prod = std::make_unique<CaloTowerConstituentsMap>( &*hcaltopo, &*cttopo );

//auto prod = std::make_unique<CaloTowerConstituentsMap>( &*hcaltopo );

  //keep geometry pointer as member for alternate EE->HE mapping
  edm::ESHandle<CaloGeometry> pG;
  iRecord.get(pG);
  const CaloGeometry* geometry = pG.product();
   
  prod->useStandardHB(true);
  if(!skipHE_) prod->useStandardHE(true);
  prod->useStandardHF(true);
  prod->useStandardHO(true);
  prod->useStandardEB(true);
   
  if (!mapFile_.empty()) {
    parseTextMap(mapFile_,*prod);
  } else if (mapAuto_ && !skipHE_) {
    assignEEtoHE(geometry, *prod, &*cttopo);
  }
  prod->sort();
  
  return prod;
}

void
CaloTowerConstituentsMapBuilder::parseTextMap( const std::string& filename, CaloTowerConstituentsMap& theMap ) {

  edm::FileInPath eff( filename );

  gzFile gzed = gzopen( eff.fullPath().c_str(), "rb" );
  
  while( !gzeof( gzed )) {
    char line[1024];
    int ieta, iphi, rawid;
    if( nullptr != gzgets( gzed, line, 1023 )) {
      if( index( line, '#' ) != nullptr )*( index( line, '#' )) = 0;
      int ct = sscanf( line, "%i %d %d", &rawid, &ieta, &iphi );
      if( ct == 3 ) {
	DetId detid( rawid );
	CaloTowerDetId tid( ieta, iphi );
	theMap.assign( detid, tid );
      }
    }
  }
  gzclose( gzed );
}

//algorithm to assign EE cells to HE towers if no text map is provided
void CaloTowerConstituentsMapBuilder::assignEEtoHE(const CaloGeometry* geometry, CaloTowerConstituentsMap& theMap, const CaloTowerTopology * cttopo){
  //get EE and HE geometries
  const CaloSubdetectorGeometry* geomEE ( geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap ) );
  if(geomEE==nullptr) return; // if no EE is defined don't know where it is used  

  const CaloSubdetectorGeometry* geomHE ( geometry->getSubdetectorGeometry( DetId::Hcal, HcalEndcap ) );
  
  //get list of EE detids
  const std::vector<DetId>& vec(geomEE->getValidDetIds());
  //loop over EE detids
  for(auto detId_itr : vec){
    //get detid position
    const CaloCellGeometry* cellGeometry = geomEE->getGeometry(detId_itr);
    const GlobalPoint& gp ( cellGeometry->getPosition() ) ;
    
    //find closest HE cell
    const HcalDetId closestCell ( geomHE->getClosestCell( gp ) ) ;
    
    //assign to appropriate CaloTower
    CaloTowerDetId tid(cttopo->convertHcaltoCT(closestCell.ietaAbs(),closestCell.subdet())*closestCell.zside(), closestCell.iphi());
    theMap.assign(detId_itr,tid);
  }
}
