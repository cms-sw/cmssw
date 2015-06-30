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

class HcalGeomParameters {

public:

  HcalGeomParameters( const DDCompactView& cpv );
  ~HcalGeomParameters();

  std::vector<double>       getConstRHB()    const {return rHB;}
  std::vector<double>       getConstDrHB()   const {return drHB;}
  std::vector<double>       getConstZHE()    const {return zHE;}
  std::vector<double>       getConstDzHE()   const {return dzHE;}
  std::vector<double>       getConstZHO()    const {return zho;}
  double                    getConstDzHF()   const {return dzVcal;}
  std::vector<double>       getConstRhoxHB() const {return rhoxb;}
  std::vector<double>       getConstZxHB()   const {return zxb;}
  std::vector<double>       getConstDyHB()   const {return dyxb;}
  std::vector<double>       getConstDxHB()   const {return dzxb;}
  std::vector<double>       getConstRhoxHE() const {return rhoxe;}
  std::vector<double>       getConstZxHE()   const {return zxe;}
  std::vector<double>       getConstDyHE()   const {return dyxe;}
  std::vector<double>       getConstDx1HE()  const {return dx1e;}
  std::vector<double>       getConstDx2HE()  const {return dx2e;}
  std::vector<int>          getConstLayHB()  const {return layb;}
  std::vector<int>          getConstLayHE()  const {return laye;}
  std::vector<double>       getConstRHO()    const;
  std::vector<int>          getModHalfHBHE(const int type) const;
       
private:
  void                initialize(const DDCompactView& cpv);
  unsigned            find (int element, std::vector<int>& array) const;
  double              getEta (double r, double z) const;
  void                loadGeometry(const DDFilteredView& _fv);

  std::vector<double> rHB, drHB;        // Radial positions of HB layers
  std::vector<double> zHE, dzHE;        // Z-positions of HE layers
  std::vector<double> zho;              // Z-positions of HO layers
  int                 nzHB, nmodHB;     // Number of halves and modules in HB
  int                 nzHE, nmodHE;     // Number of halves and modules in HE
  double              etaHO[4], rminHO; // eta in HO ring boundaries
  std::vector<double> rhoxb, zxb, dyxb, dzxb; // Geometry parameters to
  std::vector<int>    layb, laye;             // get tile size for HB & HE
  std::vector<double> zxe, rhoxe, dyxe, dx1e, dx2e; // in different layers
  double              zVcal;    // Z-position  of the front of HF
  double              dzVcal;   // Half length of the HF
  double              dlShort;  // Diference of length between long and short
};

#endif
