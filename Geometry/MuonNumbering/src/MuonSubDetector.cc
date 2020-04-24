#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"
#include <iostream>


MuonSubDetector::MuonSubDetector(const std::string& name) 
  : detectorName(name) {
  if (name=="MuonDTHits") {
    detector=barrel;
  } else if (name=="MuonCSCHits") {
    detector=endcap;
  } else if (name=="MuonRPCHits") {
    detector=rpc;
  } else if (name=="MuonGEMHits") {
    detector=gem;
  } else if (name=="MuonME0Hits") {
    detector=me0;
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

bool MuonSubDetector::isRPC(){
  return (detector==rpc);
}

bool MuonSubDetector::isGEM(){
  return (detector==gem);
}

bool MuonSubDetector::isME0(){
  return (detector==me0);
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
  } else if (detector==gem) {
    return "MuonHitsGEM";
  } else if (detector==me0) {
    return "MuonHitsME0";
  } else {
    return "";
  }
}
