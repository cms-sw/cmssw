// $Id: RecoCaloTowerCandidate.cc,v 1.2 2006/02/21 10:37:36 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

using namespace reco;

RecoCaloTowerCandidate::~RecoCaloTowerCandidate() { }

RecoCaloTowerCandidate * RecoCaloTowerCandidate::clone() const { 
  return new RecoCaloTowerCandidate( * this ); 
}

RecoCandidate::CaloTowerRef RecoCaloTowerCandidate::caloTower() const {
  return caloTower_;
}
