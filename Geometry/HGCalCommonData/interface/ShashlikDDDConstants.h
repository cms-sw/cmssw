#ifndef HGCalCommonData_ShashlikDDDConstants_h
#define HGCalCommonData_ShashlikDDDConstants_h

/** \class ShashlikDDDConstants
 *
 * this class reads the constant section of
 * the shashlik-numbering xml-file
 *  
 *  $Date: 2014/03/20 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include<string>
#include<vector>
#include<iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDCompactView;    
class DDFilteredView;

class ShashlikDDDConstants {
  
 public:
  
  ShashlikDDDConstants( const DDCompactView& cpv );
  ~ShashlikDDDConstants();
  void                loadSpecPars(const std::vector<int>& firstY,
				   const std::vector<int>& lastY);
  
  std::pair<int,int>  getSMM(int ix, int iy, bool testOnly = false) const;
  std::pair<int,int>  getXY(int sm, int mod) const;
  /// SM per side
  int                 getSuperModules() const {return 4*nSM;}
  /// modules per SM
  int                 getModules()      const {return nMods*nMods;}
  // number of SM in a row or column
  int                 getCols()         const {return 2*nRow;}
  // number of modules in a row or column
  int                 getModuleCols()         const {return getCols()*nMods;}
  
  bool                isValidXY(int ix, int iy) const;
  bool                isValidSMM(int sm, int mod) const;
  /// quadrant for module (ix:iy) 21
  ///                             34
  int                 quadrant(int ix, int iy, bool testOnly = false) const;
  int                 quadrant(int sm, bool testOnly = false) const;
  /// module ix is on the right?
  bool positiveX (int ix) const {return ix >= getModuleCols();} 
  /// module iy is on the top?
  bool positiveY (int iy) const {return iy >= getModuleCols();}
  
 private:
  void                initialize(const DDCompactView& cpv);
  void                loadSpecPars(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
				  const DDsvalues_type &) const;
  
  static const int    nMods=5;
  int                 nSM, nColS, nRow;
  std::vector<int>    firstY, lastY, firstSM, lastSM;
};

#endif
