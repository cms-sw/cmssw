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

  ShashlikDDDConstants();
  ShashlikDDDConstants( const DDCompactView& cpv );
  ~ShashlikDDDConstants();

  std::pair<int,int>  getSMM(int ix, int iy) const;
  std::pair<int,int>  getXY(int sm, int mod) const;
  int                 getSuperModules() const {return 4*nSM;}
  int                 getModules()      const {return nMods*nMods;}
  int                 getCols()         const {return 2*nRow;}
  void                initialize(const DDCompactView& cpv);
  bool                isValidXY(int ix, int iy) const;
  bool                isValidSMM(int sm, int mod) const;
  int                 quadrant(int ix, int iy) const;
  int                 quadrant(int sm) const;
       
private:
  void                checkInitialized() const;
  void                loadSpecPars(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
                                  const DDsvalues_type &) const;

  bool                tobeInitialized;
  static const int    nMods=5;
  int                 nSM, nColS, nRow;
  std::vector<int>    firstY, lastY, firstSM, lastSM;
};

#endif
