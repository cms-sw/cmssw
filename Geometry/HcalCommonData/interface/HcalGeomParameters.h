#ifndef HcalCommonData_HcalGeomParameters_h
#define HcalCommonData_HcalGeomParameters_h

/** \class HcalGeomParameters
 *
 * this class extracts some geometry constants from CompactView
 * to be used by Reco Geometry/Topology
 *  
 *  $Date: 2015/06/25 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class DDCompactView;    
class DDFilteredView;
class HcalParameters;

class HcalGeomParameters {

public:

  HcalGeomParameters();
  ~HcalGeomParameters();

  double              getConstDzHF()   const {return dzVcal;}
  void                getConstRHO( std::vector<double> & ) const;
  std::vector<int>    getModHalfHBHE(const int type) const;
  void                loadGeometry(const DDFilteredView& _fv, 
				   HcalParameters& php);

private:
  unsigned            find (int element, std::vector<int>& array) const;
  double              getEta (double r, double z) const;

  int                 nzHB, nmodHB;     // Number of halves and modules in HB
  int                 nzHE, nmodHE;     // Number of halves and modules in HE
  double              etaHO[4], rminHO; // eta in HO ring boundaries
  double              zVcal;    // Z-position  of the front of HF
  double              dzVcal;   // Half length of the HF
  double              dlShort;  // Diference of length between long and short
};

#endif
