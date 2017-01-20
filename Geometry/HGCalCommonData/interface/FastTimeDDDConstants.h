#ifndef HGCalCommonData_FastTimeDDDConstants_h
#define HGCalCommonData_FastTimeDDDConstants_h

/** \class FastTimeDDDConstants
 *
 * this class reads the constant section of
 * the numbering xml-file for fast timer device
 *  
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"

class FastTimeDDDConstants {

public:

  FastTimeDDDConstants(const FastTimeParameters* ftp);
  ~FastTimeDDDConstants();

  std::pair<int,int>  getZPhi(double z, double phi)            const;
  std::pair<int,int>  getEtaPhi(double r, double phi)          const;
  GlobalPoint         getPosition(int type, int izeta, int iphi,
				  int zside)                   const;
  std::vector<GlobalPoint> getCorners(int type, int izeta,int iphi,
				      int zside)               const;
  int                 getCells(int type)                       const;
  double              getRin(int type)                         const;
  double              getRout(int type)                        const;
  double              getZHalf(int type)                       const;
  double              getZPos(int type)                        const;
  bool                isValidXY(int type, int izeta, int iphi) const;
  int                 numberEtaZ(int type)                     const;
  int                 numberPhi(int type)                      const;
       
private:
  void                initialize();

  const FastTimeParameters* ftpar_;
  double                    etaMin_, etaMax_, dEta_;
  double                    dZBarrel_, dPhiBarrel_, dPhiEndcap_;
  std::vector<double>       rLimits_;
};

#endif
