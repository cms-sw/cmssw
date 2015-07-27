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

#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDCompactView;    
class DDFilteredView;

class FastTimeDDDConstants {

public:

  FastTimeDDDConstants( const DDCompactView& cpv );
  ~FastTimeDDDConstants();

  int                 computeCells()            const;
  int                 getType()                 const {return cellType;}
  std::pair<int,int>  getXY(int copy)           const;
  std::pair<int,int>  getXY(double x, double y) const;
  int                 getCells()                const {return 4*nCells;}
  bool                isValidXY(int ix, int iy) const;
  bool                isValidCell(int copy)     const;
  int                 quadrant(int ix, int iy)  const;
  int                 quadrant(int copy)        const;
       
private:
  void                initialize(const DDCompactView& cpv);
  void                loadSpecPars(const DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string &, 
                                  const DDsvalues_type &) const;

  int                 nCells, nCols, nRows, cellType;
  double              rIn, rOut, cellSize;
  std::vector<int>    firstY, lastY, firstCell, lastCell;
};

#endif
