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
//
//

#include "Geometry/CaloEventSetup/plugins/EcalTrigTowerConstituentsMapBuilder.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalTrigTowerConstituentsMapBuilder::EcalTrigTowerConstituentsMapBuilder(const edm::ParameterSet& iConfig)
    : mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile", "")) {
  setWhatProduced(this);
}

EcalTrigTowerConstituentsMapBuilder::~EcalTrigTowerConstituentsMapBuilder() {}

EcalTrigTowerConstituentsMapBuilder::ReturnType EcalTrigTowerConstituentsMapBuilder::produce(
    const IdealGeometryRecord& iRecord) {
  auto prod = std::make_unique<EcalTrigTowerConstituentsMap>();

  if (!mapFile_.empty()) {
    parseTextMap(mapFile_, *prod);
  }
  return prod;
}

void EcalTrigTowerConstituentsMapBuilder::parseTextMap(const std::string& filename,
                                                       EcalTrigTowerConstituentsMap& theMap) {
  edm::FileInPath eff(filename);

  std::ifstream f(eff.fullPath().c_str());
  if (!f.good())
    return;

  int ietaTower, iphiTower;
  int ix, iy, iz;
  char line[80];  // a buffer for the line to read
  char ch;        // a temporary for holding the end of line
  while ((ch = f.peek()) != '-') {
    f.get(line, 80, '\n');  // read 80 characters to end of line
    f.get(ch);              // eat out the '\n'
    // extract the numbers

    int nread = sscanf(line, " %d %d %d %d %d", &ix, &iy, &iz, &ietaTower, &iphiTower);
    if (nread == 5) {
      EEDetId eeid(ix, iy, iz, 0);
      EcalTrigTowerDetId etid(iz, EcalEndcap, ietaTower, iphiTower);
      theMap.assign(DetId(eeid), etid);
    }
  }
  // Pass comment line
  f.get(line, 80, '\n');  // read 80 characters to end of line
  f.get(ch);              // eat out the '\n'
  // Next info line
  f.get(line, 80, '\n');  // read 80 characters to end of line
  f.get(ch);              // eat out the '\n'
  f.close();
  return;
}
