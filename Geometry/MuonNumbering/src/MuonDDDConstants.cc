#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <string>
#include <iostream>
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#define LOCAL_DEBUG

MuonDDDConstants::MuonDDDConstants(){ }

int MuonDDDConstants::getValue( const std::string& name ) const {
  if ( namesAndValues_.size() == 0 ) {
    std::cout << "MuonDDDConstants::getValue HAS NO VALUES!" << std::endl;
    throw;
  }

  std::cout << "about to look for ... " << name << std::endl;
  std::map<std::string, int>::const_iterator findIt = namesAndValues_.find(name);

  if ( findIt == namesAndValues_.end() ) {
    std::cout << "MuonDDDConstants::getValue was asked for " << name << " and had NO clue!" << std::endl;
    throw;
  }

  return findIt->second;
}

void MuonDDDConstants::addValue(const std::string& name, const int& value) {
  namesAndValues_[name] = value;
}

