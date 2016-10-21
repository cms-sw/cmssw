#ifndef HGCalCommonData_FastTimeDDDConstants_h
#define HGCalCommonData_FastTimeDDDConstants_h

/** \class FastTimeDDDConstants
 *
 * this class reads the constant section of
 * the numbering xml-file for fast timer device
 *  
 *  $Date: 2014/03/20 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"
#include "G4ThreeVector.hh"

class FastTimeDDDConstants {

public:

  FastTimeDDDConstants(const FastTimeParameters* ftp);
  ~FastTimeDDDConstants();

  std::pair<int,int>  getZPhi(G4ThreeVector local)             const;
  std::pair<int,int>  getEtaPhi(G4ThreeVector local)           const;
  int                 getCells(int type)                       const;
  bool                isValidXY(int type, int izeta, int iphi) const;
       
private:
  void                initialize();

  const FastTimeParameters* ftpar_;
  double                    etaMin_, etaMax_, dEta_;
  double                    dZBarrel_, dPhiBarrel_, dPhiEndcap_;
  std::vector<double>       rLimits_;
};

#endif
