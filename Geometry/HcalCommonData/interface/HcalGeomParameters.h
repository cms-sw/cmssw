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

#include <string>
#include <vector>
#include <iostream>

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HcalCommonData/interface/HcalCellType.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HcalParameters;

class HcalGeomParameters {
public:
  HcalGeomParameters();
  ~HcalGeomParameters();

  double getConstDzHF() const { return dzVcal_; }
  void getConstRHO(std::vector<double>&) const;
  std::vector<int> getModHalfHBHE(const int type) const;
  void loadGeometry(const DDFilteredView& _fv, HcalParameters& php);
  void loadGeometry(cms::DDFilteredView& fv, HcalParameters& php);

private:
  unsigned find(int element, std::vector<int>& array) const;
  double getEta(double r, double z) const;
  void clear(HcalParameters& php);
  void loadfinal(HcalParameters& php);

  int nzHB_, nmodHB_;         // Number of halves and modules in HB
  int nzHE_, nmodHE_;         // Number of halves and modules in HE
  double etaHO_[4], rminHO_;  // eta in HO ring boundaries
  double zVcal_;              // Z-position  of the front of HF
  double dzVcal_;             // Half length of the HF
  double dlShort_;            // Diference of length between long and short
  static const int maxLayer_ = 20;
  std::vector<double> rb_, ze_, thkb_, thke_;
  std::vector<int> ib_, ie_, izb_, phib_, ize_, phie_;
  std::vector<double> rxb_, rminHE_, rmaxHE_;
};

#endif
