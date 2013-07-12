#ifndef EcalTestBeam_EcalTBCrystalMap_h
#define EcalTestBeam_EcalTBCrystalMap_h

/*
 *
 * $Id: EcalTBCrystalMap.h,v 1.2 2006/05/30 14:14:45 fabiocos Exp $
 *
 */

#include <map>
#include <iostream>
#include <fstream>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


class EcalTBCrystalMap {

 public:
  
  typedef std::map< std::pair< double, double >, int > CrystalTBIndexMap;

  EcalTBCrystalMap(std::string const & MapFileName);
  ~EcalTBCrystalMap();

  int CrystalIndex(double thisEta, double thisPhi);
  void findCrystalAngles(const int thisCrysIndex, double & thisEta, double & thisPhi);

  static const int NCRYSTAL = 1700;
  
 private:
  
  double crysEta, crysPhi;
  int crysIndex;

  CrystalTBIndexMap map_;

};

#endif
