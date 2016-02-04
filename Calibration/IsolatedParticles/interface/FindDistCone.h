#ifndef CalibrationIsolatedParticlesFindDistCone_h
#define CalibrationIsolatedParticlesFindDistCone_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <cmath>

namespace spr {

  // Cone clustering core
  double getDistInPlaneTrackDir(const GlobalPoint& caloPoint, const GlobalVector& caloVector, const GlobalPoint& rechitPoint);

  // Not used, but here for reference
  double getDistInCMatEcal(double eta1, double phi1, double eta2, double phi2);
  double getDistInCMatHcal(double eta1, double phi1, double eta2, double phi2);

  // get eta, phi, energy of rechits in collection
  void getEtaPhi(HBHERecHitCollection::const_iterator hit, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene);

  void getEtaPhi(edm::PCaloHitContainer::const_iterator hit, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene);

  void getEtaPhi(EcalRecHitCollection::const_iterator hit, std::vector<int>& RH_ieta, std::vector<int>& RH_iphi, std::vector<double>& RH_ene);

  // get eta, phi of rechits in collection
  void getEtaPhi(HBHERecHitCollection::const_iterator hit,int& ieta,int& iphi);
  void getEtaPhi(edm::PCaloHitContainer::const_iterator hit,int& ieta,int& iphi);
  void getEtaPhi(EcalRecHitCollection::const_iterator hit,int& ieta,int& iphi);

  double getEnergy(HBHERecHitCollection::const_iterator hit);
  double getEnergy(edm::PCaloHitContainer::const_iterator hit);
  double getEnergy(EcalRecHitCollection::const_iterator hit);

  GlobalPoint getGpos(const CaloGeometry* geo, HBHERecHitCollection::const_iterator hit);
  GlobalPoint getGpos(const CaloGeometry* geo, edm::PCaloHitContainer::const_iterator hit);
  GlobalPoint getGpos(const CaloGeometry* geo, EcalRecHitCollection::const_iterator hit);
  
}

#endif
