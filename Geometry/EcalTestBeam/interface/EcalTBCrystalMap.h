#ifndef EcalTestBeam_EcalTBCrystalMap_h
#define EcalTestBeam_EcalTBCrystalMap_h

/*
 *
 * $Id:$
 *
 */

#include <map>
#include <iostream>
#include <fstream>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"


class EcalTBCrystalMap {

 public:
  
  typedef std::map< std::pair< double, double >, int > CrystalTBIndexMap;

  EcalTBCrystalMap(std::string const & MapFileName);
  ~EcalTBCrystalMap();

  int CrystalIndex(double thisEta, double thisPhi);
  void findCrystalAngles(const int thisCrysIndex, double & thisEta, double & thisPhi);
  
 private:
  
  double crysEta, crysPhi;
  int crysIndex;

  CrystalTBIndexMap map_;

  static const int NCRYSTAL = 1700;

};

#endif
