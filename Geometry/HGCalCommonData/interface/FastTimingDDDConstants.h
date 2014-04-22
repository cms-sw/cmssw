#ifndef HGCalCommonData_FastTimingDDDConstants_h
#define HGCalCommonData_FastTimingDDDConstants_h

/** \class FastTimingDDDConstants
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

class FastTimingDDDConstants {

public:

  FastTimingDDDConstants();
  FastTimingDDDConstants( const DDCompactView& cpv );
  ~FastTimingDDDConstants();

  std::pair<int,int>  getXY(int copy)           const;
  std::pair<int,int>  getXY(double x, double y) const;
  int                 getCells()                const {return 4*nCells;}
  void                initialize(const DDCompactView& cpv);
  bool                isValidXY(int ix, int iy) const;
  bool                isValidCell(int copy)     const;
  int                 quadrant(int ix, int iy)  const;
  int                 quadrant(int copy)        const;
       
private:
  void                checkInitialized() const;
  void                loadSpecPars(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
                                  const DDsvalues_type &) const;

  bool                tobeInitialized;
  int                 nCells, nCols, nRows;
  double              rIn, rOut, cellSize;
  std::vector<int>    firstY, lastY, firstCell, lastCell;
};

#endif
