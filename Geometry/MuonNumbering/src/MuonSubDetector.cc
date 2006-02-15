#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"
#include <iostream>


MuonSubDetector::MuonSubDetector(std::string name) 
  : detectorName(name) {
  if (name=="MuonDTHits") {
    detector=barrel;
  } else if (name=="MuonCSCHits") {
    detector=endcap;
  } else if (name=="MuonRPCHits") {
    detector=rpc;
  } else {
    std::cout << "MuonSubDetector::MuonSubDetector does not recognize ";
    std::cout << name <<std::endl;
    detector=nodef;
  } 
}

bool MuonSubDetector::isBarrel(){
  return (detector==barrel);
}

bool MuonSubDetector::isEndcap(){
  return (detector==endcap);
}

bool MuonSubDetector::isRpc(){
  return (detector==rpc);
}

std::string MuonSubDetector::name(){
  return detectorName;
}

std::string MuonSubDetector::suIdName(){
  if (detector==barrel) {
    return "MuonHitsBarrel";
  } else if (detector==endcap) {
    return "MuonHitsEndcap";
  } else if (detector==rpc) {
    return "MuonHitsRPC";
  } else {
    return "";
  }
}
