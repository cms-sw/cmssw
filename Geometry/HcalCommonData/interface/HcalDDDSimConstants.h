#ifndef HcalCommonData_HcalDDDSimConstants_h
#define HcalCommonData_HcalDDDSimConstants_h

/** \class HcalDDDSimConstants
 *
 * this class reads the constant section of
 * the hcal-sim-numbering xml-file
 *  
 *  $Date: 2013/12/25 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
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

class HcalDDDSimConstants {

public:

  HcalDDDSimConstants();
  HcalDDDSimConstants( const DDCompactView& cpv );
  ~HcalDDDSimConstants();

  HcalCellType::HcalCell    cell(int det, int zside, int depth, int etaR, 
				 int iphi) const;
  std::vector<std::pair<double,double> > getConstHBHE(const int type) const;
  std::pair<int,double>     getDetEta(double eta, int depth);
  int                       getEta(int det,int lay, double hetaR);
  std::pair<int,int>        getEtaDepth(int det, int etaR, int phi, int depth,
					int lay);
  double                    getEtaHO(double& etaR, double& x, double& y, 
				     double& z) const;
  std::pair<int,int>        getiEtaRange(const int i) const
    {return std::pair<int,int>(etaMin[i],etaMax[i]);}
  std::vector<double>       getEtaTable() const {return etaTable;}
  std::vector<double>       getEtaTableHF() const;
  std::pair<int,int>        getModHalfHBHE(const int type) const;
  std::vector<int>          getNOff() const {return nOff;}
  double                    getPhiBin(const int i) const {return phibin[i];}
  std::pair<double,double>  getPhiCons(int det, int ieta);
  double                    getPhiOff(const int i) const {return phioff[i];}
  std::vector<double>       getPhiOffs()  const {return phioff;}
  std::vector<double>       getPhiTable() const {return phibin;}
  std::vector<double>       getPhiTableHF() const {return phitable;}
  std::vector<HcalCellType> HcalCellTypes() const;
  std::vector<HcalCellType> HcalCellTypes(HcalSubdetector, int ieta=-1,
					  int depth=-1) const;
  void                      initialize(const DDCompactView& cpv);
  int                       getMaxDepth(const int type) const {return maxDepth[type];}
  unsigned int              numberOfCells(HcalSubdetector) const;
  int                       phiNumber(int phi, int unit) const;
  void                      printTiles() const;
  int                       unitPhi(int det, int etaR) const;
  int                       unitPhi(double dphi) const; 
       
private:
  void                checkInitialized() const;
  void                loadSpecPars(const DDFilteredView& fv);
  void                loadGeometry(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
				  const DDsvalues_type &, int &) const;
  unsigned            find (int element, std::vector<int>& array) const;
  double              deltaEta(int det, int eta, int depth) const;
  double              getEta(int det, int etaR, int zside, int depth=1) const;
  double              getEta(double r, double z) const;
  int                 getShift(HcalSubdetector subdet, int depth) const;
  double              getGain (HcalSubdetector subdet, int depth) const;
  void                printTileHB(int eta, int depth) const;
  void                printTileHE(int eta, int depth) const;
  
  bool                tobeInitialized;
  static const int    nEtaMax=100;
  static const int    nDepthMax=9;
  int                 maxDepth[4]; // Maximum depths in HB/HE/HF/HO
  std::vector<double> phioff;   // Phi offset for barrel, endcap, forward
  std::vector<double> etaTable; // Eta table 
  int                 nEta;     // Number of bins in eta for HB and HE
  std::vector<double> rTable;   // R-table
  int                 nR;       // Number of bins in r
  int                 nPhi;     // Number of bins in phibin
  std::vector<double> phibin;   // Phi step for all eta bins (HB, HE and HO)
  int                 nPhiF;    // Number of bins in phitable
  std::vector<double> phitable; // Phi step for all eta bins (HF)
  std::vector<int>    layerGroup[nEtaMax]; // Depth index for a given layer
  std::vector<int>    depths[nDepthMax];   // Maximum layer number for a depth 
  int                 nDepth;   // Number of bins in depth0
  std::vector<int>    nOff;     // Speical eta bin #'s in barrel and endcap
  std::vector<int>    etaMin;   // Minimum eta bin number for HB/HE/HF
  std::vector<int>    etaMax;   // Maximum eta bin number for HB/HE/HF
  std::vector<double> etaRange; // Maximum eta value for HB/HE/HF
  std::vector<double> gparHF;   // Geometry parameters of HF
  std::vector<double> layer0Wt; // Weight of layer 0
  std::vector<double> gainHB;   // Gain factor   for HB
  std::vector<int>    shiftHB;  // Readout shift ..  ..
  std::vector<double> gainHE;   // Gain factor   for HE
  std::vector<int>    shiftHE;  // Readout shift ..  ..
  std::vector<double> gainHF;   // Gain factor   for HF
  std::vector<int>    shiftHF;  // Readout shift ..  ..

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
