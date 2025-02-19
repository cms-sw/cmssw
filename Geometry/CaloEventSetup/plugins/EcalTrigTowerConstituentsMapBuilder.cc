// -*- C++ -*-
//
// Package:    EcalTrigTowerConstituentsMapBuilder
// Class:      EcalTrigTowerConstituentsMapBuilder
// 
/**\class EcalTrigTowerConstituentsMapBuilder EcalTrigTowerConstituentsMapBuilder.h tmp/EcalTrigTowerConstituentsMapBuilder/interface/EcalTrigTowerConstituentsMapBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Paolo Meridiani
// $Id: EcalTrigTowerConstituentsMapBuilder.cc,v 1.4 2010/03/26 19:35:00 sunanda Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/EcalTrigTowerConstituentsMapBuilder.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>
#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
EcalTrigTowerConstituentsMapBuilder::EcalTrigTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig) :
  mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile",""))
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  
  //now do what ever other initialization is needed
}


EcalTrigTowerConstituentsMapBuilder::~EcalTrigTowerConstituentsMapBuilder()
{ 
}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalTrigTowerConstituentsMapBuilder::ReturnType
EcalTrigTowerConstituentsMapBuilder::produce(const IdealGeometryRecord& iRecord)
{
   std::auto_ptr<EcalTrigTowerConstituentsMap> prod(new EcalTrigTowerConstituentsMap());

   if (!mapFile_.empty()) {
     parseTextMap(mapFile_,*prod);
   }
   return prod;
}

void EcalTrigTowerConstituentsMapBuilder::parseTextMap(const std::string& filename, EcalTrigTowerConstituentsMap& theMap) {
  edm::FileInPath eff(filename);
  
  std::ifstream f(eff.fullPath().c_str());
  if (!f.good())
    return; 
  
  int ietaTower,iphiTower;
  int ix,iy,iz;
  char line[80];  // a buffer for the line to read
  char ch;        // a temporary for holding the end of line
  while ((ch = f.peek()) != '-') {
    f.get(line,80,'\n');            // read 80 characters to end of line
    f.get(ch);                      // eat out the '\n'
    // extract the numbers
/*
  int mod,cry;
    int nread = sscanf (line, " %d %d %d %d",&mod,&cry,&ietaTower,&iphiTower);
    if (nread == 4 )
      {
	EEDetId eeid(mod,cry,1,EEDetId::SCCRYSTALMODE);
	EcalTrigTowerDetId etid(1,EcalEndcap,ietaTower-45+17,iphiTower);
	//	std::cout << eeid << "\n->" << etid << std::endl;
	theMap.assign(DetId(eeid),etid);
      }
*/
    int nread = sscanf (line, " %d %d %d %d %d",&ix,&iy,&iz,&ietaTower, &iphiTower);
    if (nread == 5) {
      EEDetId eeid(ix,iy,iz,0);
      // std::cout << "-- manu ix eta phi " << DetId(eeid).rawId() << " " << iz << " " << ietaTower << " " << iphiTower << std::endl;
      EcalTrigTowerDetId etid(iz,EcalEndcap,ietaTower,iphiTower);
      theMap.assign(DetId(eeid),etid);
    }
    
  }
  // Pass comment line
  f.get(line,80,'\n');            // read 80 characters to end of line
  f.get(ch);                      // eat out the '\n'
  // Next info line
  f.get(line,80,'\n');            // read 80 characters to end of line
  f.get(ch);                      // eat out the '\n'
  // extract the numbers
  //   int nTE;
  //   sscanf (line, " %d",&nTE);
  //   nTowEta_e=nTE;
  //   while ((ch = f.peek()) != EOF) {
  //     f.get(line,80,'\n');            // read 80 characters to end of line
  //     f.get(ch);                      // eat out the '\n'
  //     // extract the numbers
  //     float bound;
  //     sscanf (line, " %f", &bound);
  //     eta_boundaries.push_back(bound);
  //   }
  
  f.close();
  return;
}
